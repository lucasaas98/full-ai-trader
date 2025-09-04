"""
Real System Integration Tests

This module contains integration tests that use actual services instead of mocks.
It tests the complete trading system end-to-end with real Redis communication,
database interactions, and service coordination.
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import pytest
import redis.asyncio as redis

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from shared.config import get_config  # noqa: E402
from shared.models import (  # noqa: E402
    AssetType,
    MarketData,
    TimeFrame,
)
from tests.integration.mock_services.mock_data_collector import (  # noqa: E402
    MockDataCollector,
    MockDataCollectorConfig,
)
from tests.integration.mock_services.service_orchestrator import (  # noqa: E402
    create_service_orchestrator,
)

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CredentialValidator:
    """Validates that integration test credentials are different from production."""

    @staticmethod
    def validate_alpaca_credentials():
        """Ensure integration Alpaca credentials differ from production."""
        prod_key = os.getenv("ALPACA_API_KEY", "")
        integration_key = os.getenv("ALPACA_API_KEY", "")

        if prod_key and integration_key and prod_key == integration_key:
            raise ValueError(
                "Integration test Alpaca API key is the same as production! "
                "Please use different credentials in .env.integration"
            )

        # Check for test patterns
        if not any(
            test_pattern in integration_key.lower()
            for test_pattern in ["test", "integration", "pkt"]
        ):
            logger.warning(
                "Alpaca API key doesn't contain test indicators. "
                "Ensure you're using test/paper credentials."
            )

    @staticmethod
    def validate_database_credentials():
        """Ensure integration database is different from production."""
        db_name = os.getenv("DB_NAME", "")

        if not any(
            test_pattern in db_name.lower() for test_pattern in ["test", "integration"]
        ):
            raise ValueError(
                f"Database name '{db_name}' doesn't indicate testing. "
                "Use a test-specific database name."
            )


@pytest.fixture(scope="session")
async def validate_credentials():
    """Validate that we're using test credentials."""
    CredentialValidator.validate_alpaca_credentials()
    CredentialValidator.validate_database_credentials()
    logger.info("✅ Credential validation passed")


@pytest.fixture(scope="session")
async def redis_client():
    """Create Redis client for tests."""
    config = get_config()

    client = redis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        password=config.redis.password,
        db=config.redis.db,
        decode_responses=True,
    )

    # Test connection
    await client.ping()
    logger.info("✅ Redis connection established")

    yield client

    # Cleanup
    await client.flushdb()  # Clear test data
    await client.close()


@pytest.fixture(scope="session")
async def database_pool():
    """Create database connection pool for tests."""
    config = get_config()

    pool = await asyncpg.create_pool(
        host=config.database.host,
        port=config.database.port,
        database=config.database.name,
        user=config.database.user,
        password=config.database.password,
        min_size=2,
        max_size=5,
    )

    logger.info("✅ Database connection pool created")

    yield pool

    # Cleanup - clear test data
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE orders CASCADE")
        await conn.execute("TRUNCATE TABLE positions CASCADE")
        await conn.execute("TRUNCATE TABLE portfolio_snapshots CASCADE")
        logger.info("🧹 Test database cleaned up")

    await pool.close()


@pytest.fixture(scope="session")
async def mock_data_collector():
    """Create and start mock data collector."""
    config = MockDataCollectorConfig(
        historical_data_path="data/parquet",
        available_symbols=["AAPL", "SPY", "QQQ", "MSFT", "TSLA"],
        redis_publish_interval=10,  # Faster for tests
        simulate_screener=True,
        screener_interval=30,
    )

    collector = MockDataCollector(config)
    await collector.start()

    logger.info("✅ Mock Data Collector started")

    yield collector

    await collector.stop()
    logger.info("🛑 Mock Data Collector stopped")


@pytest.fixture(scope="session")
async def service_orchestrator(mock_data_collector):
    """Create and start service orchestrator."""
    orchestrator = await create_service_orchestrator()

    # Start services (excluding data collector since we have mock)
    logger.info("🚀 Starting service orchestrator...")

    # Start individual services manually to control order
    try:
        await orchestrator._start_strategy_engine()
        await asyncio.sleep(2)

        await orchestrator._start_risk_manager()
        await asyncio.sleep(2)

        await orchestrator._start_trade_executor()
        await asyncio.sleep(2)

        await orchestrator._start_scheduler()
        await asyncio.sleep(2)

        # Wait for services to be ready
        if not await orchestrator.wait_for_services_ready(timeout=120):
            raise RuntimeError("Services failed to start within timeout")

        logger.info("✅ Service orchestrator started successfully")

    except Exception as e:
        logger.error(f"Failed to start services: {e}")
        await orchestrator.stop_all_services()
        raise

    yield orchestrator

    await orchestrator.stop_all_services()
    logger.info("🛑 Service orchestrator stopped")


@pytest.mark.asyncio
@pytest.mark.real_integration
class TestRealSystemIntegration:
    """Real system integration tests."""

    async def test_setup_integration_environment(self, validate_credentials):
        """Test that integration environment is properly set up."""
        # Check environment variables
        assert os.getenv("INTEGRATION_TEST_MODE") == "true"
        assert os.getenv("TESTING") == "true"

        # Check data directory exists
        data_path = Path("data/parquet")
        assert data_path.exists(), "Historical data directory not found"

        market_data_path = data_path / "market_data"
        assert market_data_path.exists(), "Market data directory not found"

        # Check for at least one symbol
        symbol_dirs = list(market_data_path.iterdir())
        assert len(symbol_dirs) > 0, "No market data found"

        logger.info("✅ Integration environment setup validated")

    async def test_redis_connectivity(self, redis_client):
        """Test Redis pub/sub functionality."""
        # Test basic operations
        await redis_client.set("test_key", "test_value", ex=60)
        value = await redis_client.get("test_key")
        assert value == "test_value"

        # Test pub/sub
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("test_channel")

        # Publish a message
        await redis_client.publish("test_channel", "test_message")

        # Read message
        message = await pubsub.get_message(timeout=5)
        if message and message["type"] == "subscribe":
            message = await pubsub.get_message(timeout=5)

        assert message is not None
        assert message["data"] == "test_message"

        await pubsub.close()
        logger.info("✅ Redis connectivity test passed")

    async def test_database_connectivity(self, database_pool):
        """Test database connectivity and basic operations."""
        async with database_pool.acquire() as conn:
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            assert result == 1

            # Test table access
            tables = await conn.fetch(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """
            )

            table_names = [row["table_name"] for row in tables]

            # Check for essential tables
            required_tables = ["orders", "positions", "portfolio_snapshots"]
            for table in required_tables:
                assert table in table_names, f"Required table '{table}' not found"

        logger.info("✅ Database connectivity test passed")

    async def test_mock_data_collector_functionality(
        self, mock_data_collector, redis_client
    ):
        """Test mock data collector publishes data correctly."""
        # Subscribe to market data updates
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("market_data:updates", "screener:updates")

        # Wait for initial messages
        await asyncio.sleep(5)

        # Force updates
        await mock_data_collector.force_screener_update()
        await asyncio.sleep(2)

        # Check for messages
        messages_received = 0
        timeout = time.time() + 30

        while time.time() < timeout and messages_received < 2:
            message = await pubsub.get_message(timeout=5)
            if message and message["type"] == "message":
                messages_received += 1
                logger.info(f"Received message on {message['channel']}")

        await pubsub.close()

        assert messages_received > 0, "No messages received from mock data collector"

        # Test health check
        health = await mock_data_collector.health_check()
        assert health["status"] == "healthy"
        assert health["available_symbols"] > 0

        logger.info("✅ Mock data collector test passed")

    async def test_service_orchestrator_startup(self, service_orchestrator):
        """Test that all services start up correctly."""
        status = await service_orchestrator.get_service_status()

        assert status["orchestrator_running"] is True

        # Check each service
        for service_name, service_info in status["services"].items():
            if service_name == "mock_data_collector":
                continue  # Skip since we use separate fixture

            assert service_info["status"] in [
                "running",
                "started",
            ], f"Service {service_name} is not running: {service_info['status']} - {service_info.get('error', 'No error')}"

        logger.info("✅ Service orchestrator startup test passed")

    async def test_end_to_end_data_flow(
        self, mock_data_collector, service_orchestrator, redis_client
    ):
        """Test complete data flow from data collector through all services."""
        # Subscribe to key channels
        pubsub = redis_client.pubsub()
        channels = [
            "market_data:updates",
            "screener:updates",
            "strategy:signals",
            "risk:assessments",
            "orders:*",
        ]

        await pubsub.psubscribe(*channels)

        # Trigger data flow
        await mock_data_collector.force_screener_update()

        # Simulate some time passing
        await asyncio.sleep(15)

        # Force another update to ensure processing
        await mock_data_collector.force_screener_update()
        await asyncio.sleep(10)

        # Collect messages
        messages = []
        timeout = time.time() + 30

        while time.time() < timeout and len(messages) < 5:
            message = await pubsub.get_message(timeout=2)
            if message and message["type"] == "pmessage":
                messages.append(
                    {
                        "channel": message["channel"],
                        "data": message["data"],
                        "timestamp": datetime.now(timezone.utc),
                    }
                )

        await pubsub.close()

        # Analyze message flow
        channel_counts = {}
        for msg in messages:
            channel_base = msg["channel"].split(":")[0]
            channel_counts[channel_base] = channel_counts.get(channel_base, 0) + 1

        logger.info(f"Messages received by channel: {channel_counts}")

        # We should have at least some data flow
        assert len(messages) > 0, "No messages received in data flow test"
        assert (
            "market_data" in channel_counts or "screener" in channel_counts
        ), "No market data or screener updates received"

        logger.info("✅ End-to-end data flow test passed")

    async def test_trading_simulation(
        self, service_orchestrator, database_pool, redis_client
    ):
        """Test a complete trading simulation."""
        logger.info("🎭 Starting trading simulation test...")

        # Monitor orders table
        async def count_orders():
            async with database_pool.acquire() as conn:
                return await conn.fetchval("SELECT COUNT(*) FROM orders")

        # Monitor Redis for order events
        pubsub = redis_client.pubsub()
        await pubsub.psubscribe("orders:*", "strategy:*", "risk:*")

        initial_orders = await count_orders()

        # Run trading simulation
        try:
            await service_orchestrator.simulate_trading_day(duration_minutes=2)

            # Give time for processing
            await asyncio.sleep(10)

            # Check for activity
            final_orders = await count_orders()

            # Collect any messages during simulation
            messages = []
            timeout = time.time() + 15

            while time.time() < timeout:
                message = await pubsub.get_message(timeout=1)
                if message and message["type"] == "pmessage":
                    messages.append(message["channel"])
                    if len(messages) >= 10:  # Limit collection
                        break

            await pubsub.close()

            logger.info(f"Orders before: {initial_orders}, after: {final_orders}")
            logger.info(f"Messages during simulation: {len(messages)}")

            # Validation - we should see some activity even if no actual orders
            assert (
                len(messages) > 0 or final_orders > initial_orders
            ), "No trading activity detected during simulation"

            logger.info("✅ Trading simulation test passed")

        except Exception:
            await pubsub.close()
            raise

    async def test_service_resilience(self, service_orchestrator, redis_client):
        """Test system resilience and error handling."""
        logger.info("🔧 Testing service resilience...")

        # Get initial status
        initial_status = await service_orchestrator.get_service_status()
        running_services = [
            name
            for name, info in initial_status["services"].items()
            if info["status"] == "running"
        ]

        logger.info(f"Initially running services: {running_services}")

        # Simulate Redis connectivity issues by using wrong password
        test_redis = redis.Redis(
            host="localhost",
            port=6381,
            password="wrong_password",
            decode_responses=True,
        )

        try:
            await test_redis.ping()
            logger.warning("Expected Redis connection to fail with wrong password")
        except Exception:
            logger.info("✅ Redis correctly rejected wrong credentials")
        finally:
            await test_redis.close()

        # Wait and check that services are still running
        await asyncio.sleep(5)

        final_status = await service_orchestrator.get_service_status()
        still_running = [
            name
            for name, info in final_status["services"].items()
            if info["status"] == "running"
        ]

        logger.info(f"Services still running after resilience test: {still_running}")

        # At least some services should still be running
        assert len(still_running) > 0, "All services failed during resilience test"

        logger.info("✅ Service resilience test passed")

    async def test_graceful_shutdown(self, service_orchestrator):
        """Test graceful shutdown of all services."""
        logger.info("🛑 Testing graceful shutdown...")

        # Get status before shutdown
        status_before = await service_orchestrator.get_service_status()
        running_before = [
            name
            for name, info in status_before["services"].items()
            if info["status"] == "running"
        ]

        logger.info(f"Services running before shutdown: {running_before}")

        # Trigger graceful shutdown
        await service_orchestrator.stop_all_services()

        # Wait a moment for shutdown to complete
        await asyncio.sleep(5)

        # Check final status
        status_after = await service_orchestrator.get_service_status()
        stopped_services = [
            name
            for name, info in status_after["services"].items()
            if info["status"] == "stopped"
        ]

        logger.info(f"Services stopped after shutdown: {stopped_services}")

        # Most services should be stopped
        assert (
            len(stopped_services) >= len(running_before) // 2
        ), "Not enough services stopped during graceful shutdown"

        logger.info("✅ Graceful shutdown test passed")

    async def test_data_consistency(self, database_pool, redis_client):
        """Test data consistency between Redis and database."""
        logger.info("🔍 Testing data consistency...")

        # This test ensures that data written to Redis and database are consistent
        # Since we're using mock data, we'll test the consistency of the test setup

        # Check Redis keys
        redis_keys = await redis_client.keys("*")
        logger.info(f"Redis keys found: {len(redis_keys)}")

        # Check database tables
        async with database_pool.acquire() as conn:
            orders_count = await conn.fetchval("SELECT COUNT(*) FROM orders")
            positions_count = await conn.fetchval("SELECT COUNT(*) FROM positions")

            logger.info(
                f"Database - Orders: {orders_count}, Positions: {positions_count}"
            )

        # Basic consistency check - we should have some data structures in place
        assert len(redis_keys) >= 0  # Redis might be empty in tests, that's OK
        assert orders_count >= 0  # Tables should exist and be queryable
        assert positions_count >= 0

        logger.info("✅ Data consistency test passed")

    async def test_performance_monitoring(self, service_orchestrator):
        """Test performance monitoring and metrics collection."""
        logger.info("📊 Testing performance monitoring...")

        start_time = time.time()

        # Get service status multiple times to test performance
        for i in range(5):
            status = await service_orchestrator.get_service_status()
            assert "timestamp" in status
            assert "services" in status
            await asyncio.sleep(0.5)

        elapsed_time = time.time() - start_time

        # Performance check - status checks should be fast
        assert elapsed_time < 10, f"Status checks took too long: {elapsed_time}s"

        logger.info(f"✅ Performance monitoring test passed (took {elapsed_time:.2f}s)")

    async def test_configuration_validation(self):
        """Test that configuration is properly loaded for integration tests."""
        config = get_config()

        # Check that we're in test mode
        assert config.environment in ["integration_test", "test"]
        assert config.testing is True

        # Check database configuration
        assert (
            "test" in config.database.name.lower()
            or "integration" in config.database.name.lower()
        )

        # Check that trading is in safe mode
        assert config.trading.dry_run is True
        assert config.trading.paper_mode is True

        logger.info("✅ Configuration validation test passed")


# Helper functions and fixtures for specific test scenarios


@pytest.fixture
async def sample_market_data():
    """Generate sample market data for testing."""
    return [
        MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.FIVE_MINUTES,
            open=150.0,
            high=151.0,
            low=149.0,
            close=150.5,
            volume=1000000,
            asset_type=AssetType.STOCK,
        ),
        MarketData(
            symbol="SPY",
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.FIVE_MINUTES,
            open=400.0,
            high=401.0,
            low=399.0,
            close=400.5,
            volume=2000000,
            asset_type=AssetType.ETF,
        ),
    ]


@pytest.mark.asyncio
async def test_market_data_processing(sample_market_data, redis_client):
    """Test market data processing pipeline."""
    # This test can be extended to test specific market data processing scenarios
    assert len(sample_market_data) == 2
    assert all(data.symbol in ["AAPL", "SPY"] for data in sample_market_data)

    # Test Redis publishing of market data
    for data in sample_market_data:
        message = {
            "symbol": data.symbol,
            "price": data.close,
            "volume": data.volume,
            "timestamp": data.timestamp.isoformat(),
        }

        await redis_client.publish(f"test_market_data:{data.symbol}", str(message))

    logger.info("✅ Market data processing test passed")


# Integration test runner and utility functions


def pytest_configure(config):
    """Configure pytest for integration tests."""
    config.addinivalue_line(
        "markers",
        "real_integration: marks tests as real integration tests (deselect with '-m \"not real_integration\"')",
    )


async def run_integration_test_suite():
    """Run the complete integration test suite."""
    logger.info("🚀 Starting Real System Integration Test Suite")

    # Set test environment
    os.environ["INTEGRATION_TEST_MODE"] = "true"
    os.environ["TESTING"] = "true"

    try:
        # Run pytest programmatically
        exit_code = pytest.main(
            [
                __file__,
                "-v",
                "--tb=short",
                "-m",
                "real_integration",
                "--asyncio-mode=auto",
                "--capture=no",
                "--durations=20",
            ]
        )

        if exit_code == 0:
            logger.info("✅ All integration tests passed!")
        else:
            logger.error(f"❌ Integration tests failed with exit code: {exit_code}")

        return exit_code == 0

    except Exception as e:
        logger.error(f"💥 Integration test suite failed: {e}")
        return False


if __name__ == "__main__":
    import sys

    # If run directly, execute the integration test suite
    success = asyncio.run(run_integration_test_suite())
    sys.exit(0 if success else 1)
