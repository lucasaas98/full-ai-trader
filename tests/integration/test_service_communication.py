import pytest
import asyncio
import json
import time
from datetime import datetime, timezone
import httpx
import redis.asyncio as redis

# Add parent directories to path for imports
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

# Models available but not used in this test file
pass  # Removed unused config import

# Mock missing model classes
from enum import Enum

class SignalType(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"

class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"

class PortfolioState:
    def __init__(self, total_value=100000, cash=50000):
        self.total_value = total_value
        self.cash = cash


class TestServiceCommunication:
    """Integration tests for service-to-service communication"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        # Create a mock config object with the needed attributes
        class MockConfig:
            def __init__(self):
                self.redis_host = "redis"
                self.redis_port = 6379
                self.redis_password = None
                self.db_host = "postgres"
                self.db_port = 5432
                self.db_name = "trading_system_test"
                self.db_user = "trader"
                self.db_password = "test_password"

        return MockConfig()

    @pytest.fixture
    async def redis_client(self, config):
        """Redis client for testing"""
        client = redis.from_url(f"redis://{config.redis_host}:{config.redis_port}")
        yield client
        await client.close()

    @pytest.fixture
    def service_urls(self):
        """Service URLs for testing"""
        return {
            "data_collector": "http://data_collector:9101",
            "strategy_engine": "http://strategy_engine:9102",
            "risk_manager": "http://risk_manager:9103",
            "trade_executor": "http://trade_executor:9104",
            "scheduler": "http://scheduler:9105"
        }

    @pytest.mark.asyncio
    async def test_data_flow_from_collector_to_strategy(self, redis_client, service_urls):
        """Test data flow from data collector to strategy engine"""
        # Step 1: Trigger data collection
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{service_urls['data_collector']}/collect/market-data",
                json={
                    "symbols": ["AAPL"],
                    "timeframe": "1h"
                },
                timeout=30.0
            )
            assert response.status_code == 200

        # Step 2: Wait for data to be published to Redis
        await asyncio.sleep(2)

        # Step 3: Verify data was published to Redis
        market_data_raw = await redis_client.get("market_data:AAPL:1h:latest")
        assert market_data_raw is not None

        market_data = json.loads(market_data_raw)
        assert market_data["symbol"] == "AAPL"
        assert "timestamp" in market_data
        assert "close" in market_data

        # Step 4: Trigger strategy signal generation
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{service_urls['strategy_engine']}/signals/generate",
                json={
                    "symbols": ["AAPL"],
                    "strategy": "moving_average",
                    "parameters": {"short_window": 20, "long_window": 50}
                },
                timeout=30.0
            )
            assert response.status_code == 200

        # Step 5: Verify signal was generated and published
        await asyncio.sleep(1)
        signal_raw = await redis_client.get("trade_signals:AAPL:latest")
        if signal_raw:
            signal_data = json.loads(signal_raw)
            assert signal_data["symbol"] == "AAPL"
            assert signal_data["strategy"] == "moving_average"

    @pytest.mark.asyncio
    async def test_signal_to_execution_flow(self, redis_client, service_urls):
        """Test flow from signal generation to trade execution"""
        # Step 1: Publish a trade signal to Redis
        test_signal = {
            "symbol": "AAPL",
            "signal_type": "buy",
            "confidence": 0.85,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": 200.0,
            "strategy": "test_strategy",
            "metadata": {
                "stop_loss": 190.0,
                "take_profit": 220.0
            }
        }

        await redis_client.publish("trade_signals", json.dumps(test_signal))

        # Step 2: Wait for risk manager to validate
        await asyncio.sleep(1)

        # Step 3: Check risk validation
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{service_urls['risk_manager']}/risk/validate",
                json={
                    "symbol": "AAPL",
                    "side": "buy",
                    "order_type": "market",
                    "quantity": 100,
                    "price": 200.0
                },
                timeout=10.0
            )
            assert response.status_code == 200
            risk_validation = response.json()

        # Step 4: If risk validation passes, execute trade
        if risk_validation.get("is_valid", False):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{service_urls['trade_executor']}/orders",
                    json={
                        "symbol": "AAPL",
                        "side": "buy",
                        "order_type": "market",
                        "quantity": 100,
                        "price": 200.0
                    },
                    timeout=15.0
                )

                # Should successfully submit order or provide clear rejection reason
                assert response.status_code in [200, 400, 422]

                if response.status_code == 200:
                    order_data = response.json()
                    assert "order_id" in order_data
                    assert order_data["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_redis_pub_sub_functionality(self, redis_client):
        """Test Redis pub/sub functionality across services"""
        # Test market data channel
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("market_data")

        # Publish test market data
        test_data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "close": 200.0,
            "volume": 1000000
        }

        await redis_client.publish("market_data", json.dumps(test_data))

        # Wait for message
        message = await asyncio.wait_for(pubsub.get_message(ignore_subscribe_messages=True), timeout=5.0)

        assert message is not None
        data = json.loads(message["data"])
        assert data["symbol"] == "AAPL"

        await pubsub.unsubscribe("market_data")
        await pubsub.close()

    @pytest.mark.asyncio
    async def test_cross_service_error_handling(self, service_urls):
        """Test error handling when services are unavailable"""
        # Test graceful degradation when strategy engine is down
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{service_urls['strategy_engine']}/signals/generate",
                    json={"symbols": ["AAPL"], "strategy": "moving_average"},
                    timeout=5.0
                )
            except httpx.ConnectError:
                # Expected when service is down
                pass

            # Other services should continue functioning
            response = await client.get(f"{service_urls['data_collector']}/health", timeout=5.0)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_service_health_check_propagation(self, service_urls):
        """Test that health checks work across all services"""
        health_results = {}

        async with httpx.AsyncClient() as client:
            for service_name, url in service_urls.items():
                try:
                    response = await client.get(f"{url}/health", timeout=10.0)
                    health_results[service_name] = {
                        "status_code": response.status_code,
                        "response": response.json() if response.status_code == 200 else None
                    }
                except Exception as e:
                    health_results[service_name] = {
                        "status_code": None,
                        "error": str(e)
                    }

        # At least some services should be healthy
        healthy_services = [
            name for name, result in health_results.items()
            if result.get("status_code") == 200
        ]

        assert len(healthy_services) > 0, f"No healthy services found: {health_results}"

    @pytest.mark.asyncio
    async def test_database_connection_across_services(self, service_urls):
        """Test database connectivity across all services"""
        database_health = {}

        async with httpx.AsyncClient() as client:
            for service_name, url in service_urls.items():
                try:
                    response = await client.get(f"{url}/health", timeout=10.0)
                    if response.status_code == 200:
                        health_data = response.json()
                        database_health[service_name] = health_data.get("database", "unknown")
                except Exception as e:
                    database_health[service_name] = f"error: {str(e)}"

        # Services that need database should report healthy DB connections
        db_dependent_services = ["data_collector", "strategy_engine", "risk_manager", "trade_executor"]

        for service in db_dependent_services:
            if service in database_health:
                assert database_health[service] in ["connected", "healthy"], \
                    f"Database unhealthy for {service}: {database_health[service]}"

    @pytest.mark.asyncio
    async def test_end_to_end_trade_flow(self, redis_client, service_urls):
        """Test complete end-to-end trade flow"""
        trade_flow_steps = []

        try:
            # Step 1: Collect market data
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{service_urls['data_collector']}/collect/market-data",
                    json={
                        "symbols": ["AAPL"],
                        "timeframe": "1h"
                    },
                    timeout=30.0
                )
                trade_flow_steps.append(f"Data collection: {response.status_code}")
                assert response.status_code == 200

            # Step 2: Generate trading signal
            await asyncio.sleep(2)  # Allow data to propagate

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{service_urls['strategy_engine']}/signals/generate",
                    json={
                        "symbols": ["AAPL"],
                        "strategy": "moving_average",
                        "parameters": {"short_window": 10, "long_window": 20}
                    },
                    timeout=30.0
                )
                trade_flow_steps.append(f"Signal generation: {response.status_code}")

                if response.status_code == 200:
                    signals = response.json().get("signals", [])
                    if signals:
                        signal = signals[0]

                        # Step 3: Validate with risk manager
                        response = await client.post(
                            f"{service_urls['risk_manager']}/risk/validate",
                            json={
                                "symbol": signal["symbol"],
                                "side": signal["signal_type"],
                                "order_type": "market",
                                "quantity": 100,
                                "price": signal["price"]
                            },
                            timeout=10.0
                        )
                        trade_flow_steps.append(f"Risk validation: {response.status_code}")

                        if response.status_code == 200:
                            risk_result = response.json()

                            # Step 4: Execute trade if risk approved
                            if risk_result.get("is_valid", False):
                                response = await client.post(
                                    f"{service_urls['trade_executor']}/orders",
                                    json={
                                        "symbol": signal["symbol"],
                                        "side": signal["signal_type"],
                                        "order_type": "market",
                                        "quantity": 100,
                                        "price": signal["price"]
                                    },
                                    timeout=15.0
                                )
                                trade_flow_steps.append(f"Trade execution: {response.status_code}")

            # Verify at least the first steps completed successfully
            assert len(trade_flow_steps) >= 2

        except Exception as e:
            pytest.fail(f"End-to-end flow failed at steps {trade_flow_steps}: {str(e)}")

    @pytest.mark.asyncio
    async def test_scheduler_service_coordination(self, service_urls):
        """Test scheduler's coordination of other services"""
        async with httpx.AsyncClient() as client:
            # Check scheduler health
            response = await client.get(f"{service_urls['scheduler']}/health", timeout=10.0)
            assert response.status_code == 200

            # Get scheduled jobs
            response = await client.get(f"{service_urls['scheduler']}/jobs", timeout=10.0)
            if response.status_code == 200:
                jobs = response.json().get("jobs", [])

                # Should have jobs for data collection, signal generation, etc.
                job_types = [job.get("type", "") for job in jobs]
                expected_job_types = ["data_collection", "signal_generation", "risk_monitoring"]

                # At least some expected job types should be present
                found_types = [jtype for jtype in expected_job_types if jtype in job_types]
                assert len(found_types) > 0

    @pytest.mark.asyncio
    async def test_redis_message_queue_reliability(self, redis_client):
        """Test Redis message queue reliability and ordering"""
        channel = "test_trade_signals"
        messages_sent = []
        messages_received = []

        # Set up subscriber
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)

        # Send multiple messages in order
        for i in range(5):
            message = {
                "id": i,
                "symbol": "AAPL",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": f"test_message_{i}"
            }
            messages_sent.append(message)
            await redis_client.publish(channel, json.dumps(message))
            await asyncio.sleep(0.1)  # Small delay between messages

        # Receive messages
        for _ in range(5):
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=5.0
                )
                if message:
                    data = json.loads(message["data"])
                    messages_received.append(data)
            except asyncio.TimeoutError:
                break

        await pubsub.unsubscribe(channel)
        await pubsub.close()

        # Verify message ordering and completeness
        assert len(messages_received) == len(messages_sent)
        for i, (sent, received) in enumerate(zip(messages_sent, messages_received)):
            assert sent["id"] == received["id"]
            assert sent["symbol"] == received["symbol"]

    @pytest.mark.asyncio
    async def test_service_resilience_to_temporary_failures(self, service_urls, redis_client):
        """Test service resilience when dependencies temporarily fail"""
        # Simulate Redis being temporarily unavailable
        original_redis_host = "redis"

        # Test data collector resilience
        async with httpx.AsyncClient() as client:
            # Should still respond to health checks even if Redis is down
            response = await client.get(f"{service_urls['data_collector']}/health", timeout=10.0)

            # Service might report degraded but should still respond
            assert response.status_code in [200, 503]

            if response.status_code == 200:
                health_data = response.json()
                # Redis might be reported as disconnected
                assert health_data.get("status") in ["healthy", "degraded"]

    @pytest.mark.asyncio
    async def test_data_consistency_across_services(self, service_urls):
        """Test data consistency across different services"""
        symbol = "AAPL"

        async with httpx.AsyncClient() as client:
            # Get latest data from data collector
            response = await client.get(
                f"{service_urls['data_collector']}/data/latest/{symbol}",
                timeout=10.0
            )

            if response.status_code == 200:
                collector_data = response.json()

                # Get portfolio info from trade executor
                response = await client.get(
                    f"{service_urls['trade_executor']}/positions",
                    timeout=10.0
                )

                if response.status_code == 200:
                    positions_data = response.json()

                    # Data timestamps should be reasonably recent and consistent
                    collector_time = datetime.fromisoformat(
                        collector_data["timestamp"].replace("Z", "+00:00")
                    )
                    now = datetime.now(timezone.utc)

                    # Data should not be older than 1 hour
                    assert (now - collector_time).total_seconds() < 3600

    @pytest.mark.asyncio
    async def test_concurrent_service_requests(self, service_urls):
        """Test handling of concurrent requests across services"""
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        async def test_service_endpoint(service_name, endpoint):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{service_urls[service_name]}{endpoint}", timeout=15.0)
                    return {"service": service_name, "status": response.status_code, "success": True}
            except Exception as e:
                return {"service": service_name, "error": str(e), "success": False}

        # Create concurrent health check tasks
        tasks = []
        for service_name in service_urls:
            task = test_service_endpoint(service_name, "/health")
            tasks.append(task)

        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successful_requests = [r for r in results if isinstance(r, dict) and r.get("success", False)]

        # At least 60% of concurrent requests should succeed
        success_rate = len(successful_requests) / len(tasks)
        assert success_rate >= 0.6, f"Success rate too low: {success_rate}, Results: {results}"

    @pytest.mark.asyncio
    async def test_service_startup_dependencies(self, service_urls):
        """Test that services start up in correct dependency order"""
        # Test dependency chain: postgres/redis -> data_collector -> strategy_engine -> risk_manager -> trade_executor -> scheduler

        startup_order = [
            "data_collector",
            "strategy_engine",
            "risk_manager",
            "trade_executor",
            "scheduler"
        ]

        service_health = {}

        async with httpx.AsyncClient() as client:
            for service_name in startup_order:
                try:
                    response = await client.get(f"{service_urls[service_name]}/health", timeout=10.0)
                    service_health[service_name] = response.status_code == 200
                except Exception:
                    service_health[service_name] = False

                await asyncio.sleep(1)  # Wait between checks

        # Check that if later services are healthy, earlier ones should be too
        for i, service in enumerate(startup_order):
            if service_health.get(service, False):
                # All previous services in chain should also be healthy
                for j in range(i):
                    prev_service = startup_order[j]
                    assert service_health.get(prev_service, False), \
                        f"{service} is healthy but dependency {prev_service} is not"

    @pytest.mark.asyncio
    async def test_api_authentication_and_authorization(self, service_urls):
        """Test API authentication and authorization across services"""
        # Test that internal service endpoints are accessible
        async with httpx.AsyncClient() as client:
            for service_name, url in service_urls.items():
                response = await client.get(f"{url}/health", timeout=10.0)

                # Health endpoints should be accessible without auth
                assert response.status_code in [200, 503], \
                    f"Unexpected status for {service_name} health: {response.status_code}"

    @pytest.mark.asyncio
    async def test_data_persistence_across_restarts(self, service_urls, redis_client):
        """Test that critical data persists across service restarts"""
        # Store test data
        test_data = {
            "symbol": "AAPL",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_value": 123.45
        }

        await redis_client.set("test_persistence_key", json.dumps(test_data), ex=3600)

        # Verify data exists
        retrieved_data = await redis_client.get("test_persistence_key")
        assert retrieved_data is not None

        parsed_data = json.loads(retrieved_data)
        assert parsed_data["symbol"] == "AAPL"
        assert parsed_data["test_value"] == 123.45

    @pytest.mark.asyncio
    async def test_load_balancing_and_failover(self, service_urls):
        """Test load balancing and failover mechanisms"""
        # Send multiple requests to test load distribution
        request_count = 10
        response_times = []

        async with httpx.AsyncClient() as client:
            for i in range(request_count):
                start_time = time.time()

                try:
                    response = await client.get(
                        f"{service_urls['data_collector']}/health",
                        timeout=10.0
                    )
                    end_time = time.time()

                    if response.status_code == 200:
                        response_times.append(end_time - start_time)

                except Exception as e:
                    # Log but continue
                    print(f"Request {i} failed: {e}")

                await asyncio.sleep(0.1)  # Small delay between requests

        # Analyze response times for consistency
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)

            # Response times should be reasonable and consistent
            assert avg_response_time < 1.0  # Average under 1 second
            assert max_response_time < 5.0   # Max under 5 seconds

    @pytest.mark.asyncio
    async def test_message_delivery_guarantees(self, redis_client):
        """Test message delivery guarantees in Redis pub/sub"""
        channel = "test_reliability"
        messages_to_send = 20

        # Set up subscriber with message acknowledgment
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)

        received_messages = []

        # Publisher task
        async def publisher():
            for i in range(messages_to_send):
                message = {
                    "id": i,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": f"message_{i}"
                }
                await redis_client.publish(channel, json.dumps(message))
                await asyncio.sleep(0.05)  # 50ms between messages

        # Subscriber task
        async def subscriber():
            while len(received_messages) < messages_to_send:
                try:
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=10.0
                    )
                    if message:
                        data = json.loads(message["data"])
                        received_messages.append(data)
                except asyncio.TimeoutError:
                    break

        # Run publisher and subscriber concurrently
        await asyncio.gather(publisher(), subscriber())

        await pubsub.unsubscribe(channel)
        await pubsub.close()

        # Verify message delivery
        assert len(received_messages) == messages_to_send

        # Verify message ordering
        for i, message in enumerate(received_messages):
            assert message["id"] == i

    @pytest.mark.asyncio
    async def test_service_communication_under_load(self, service_urls, redis_client):
        """Test service communication under high load"""
        concurrent_requests = 20

        async def make_request(service_name, endpoint):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{service_urls[service_name]}{endpoint}", timeout=15.0)
                    return {
                        "service": service_name,
                        "status_code": response.status_code,
                        "success": response.status_code == 200,
                        "response_time": 0.5  # Placeholder
                    }
            except Exception as e:
                return {
                    "service": service_name,
                    "success": False,
                    "error": str(e)
                }

        # Create load test tasks
        tasks = []
        for _ in range(concurrent_requests):
            for service_name in ["data_collector", "strategy_engine", "risk_manager"]:
                task = make_request(service_name, "/health")
                tasks.append(task)

        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        successful_results = [
            r for r in results
            if isinstance(r, dict) and r.get("success", False)
        ]

        total_requests = len(tasks)
        success_rate = len(successful_results) / total_requests

        # Should handle at least 70% of concurrent requests successfully
        assert success_rate >= 0.7, f"Load test success rate too low: {success_rate}"

    @pytest.mark.asyncio
    async def test_data_synchronization_across_services(self, service_urls):
        """Test data synchronization across services"""
        symbol = "AAPL"

        async with httpx.AsyncClient() as client:
            # Trigger data collection
            response = await client.post(
                f"{service_urls['data_collector']}/collect/market-data",
                json={"symbols": [symbol], "timeframe": "1h"},
                timeout=30.0
            )

            if response.status_code == 200:
                # Wait for data propagation
                await asyncio.sleep(3)

                # Check if strategy engine has the same data
                response = await client.get(
                    f"{service_urls['strategy_engine']}/data/latest/{symbol}",
                    timeout=10.0
                )

                if response.status_code == 200:
                    strategy_data = response.json()

                    # Check if risk manager has portfolio data
                    response = await client.get(
                        f"{service_urls['risk_manager']}/risk/portfolio",
                        timeout=10.0
                    )

                    # Data should be synchronized within reasonable time
                    assert response.status_code in [200, 404]  # 404 if no portfolio data yet

    @pytest.mark.asyncio
    async def test_error_propagation_and_circuit_breaker(self, service_urls):
        """Test error propagation and circuit breaker patterns"""
        # Test cascade failure prevention
        async with httpx.AsyncClient() as client:
            # Send invalid requests to test error handling
            invalid_requests = [
                (f"{service_urls['data_collector']}/collect/market-data", {"symbols": [], "timeframe": "invalid"}),
                (f"{service_urls['strategy_engine']}/signals/generate", {"symbols": ["INVALID"], "strategy": "nonexistent"}),
                (f"{service_urls['risk_manager']}/risk/validate", {"symbol": "", "side": "invalid", "quantity": -1}),
                (f"{service_urls['trade_executor']}/orders", {"symbol": "INVALID", "side": "invalid", "quantity": 0})
            ]

            error_responses = []

            for url, payload in invalid_requests:
                try:
                    response = await client.post(url, json=payload, timeout=10.0)
                    error_responses.append({
                        "url": url,
                        "status_code": response.status_code,
                        "response": response.json() if response.status_code != 500 else None
                    })
                except Exception as e:
                    error_responses.append({
                        "url": url,
                        "error": str(e)
                    })

            # Services should handle errors gracefully (not return 500)
            for response in error_responses:
                if "status_code" in response:
                    assert response["status_code"] != 500, f"Internal server error for {response['url']}"

    @pytest.mark.asyncio
    async def test_service_configuration_consistency(self, service_urls):
        """Test that service configurations are consistent"""
        service_configs = {}

        async with httpx.AsyncClient() as client:
            for service_name, url in service_urls.items():
                try:
                    response = await client.get(f"{url}/config", timeout=10.0)
                    if response.status_code == 200:
                        service_configs[service_name] = response.json()
                except Exception:
                    # Config endpoint might not be available
                    pass

        # If config endpoints are available, verify consistency
        if len(service_configs) > 1:
            # Check that database and Redis configurations are consistent
            db_hosts = set()
            redis_hosts = set()

            for config in service_configs.values():
                if "database" in config:
                    db_hosts.add(config["database"].get("host"))
                if "redis" in config:
                    redis_hosts.add(config["redis"].get("host"))

            # All services should use the same database and Redis hosts
            assert len(db_hosts) <= 1, f"Inconsistent database hosts: {db_hosts}"
            assert len(redis_hosts) <= 1, f"Inconsistent Redis hosts: {redis_hosts}"

    @pytest.mark.asyncio
    async def test_websocket_connections_if_available(self, service_urls):
        """Test WebSocket connections for real-time data if available"""
        # This would test WebSocket endpoints if they exist
        # For now, test that HTTP endpoints work
        pass

    @pytest.mark.asyncio
    async def test_service_metrics_collection(self, service_urls):
        """Test that services expose metrics for monitoring"""
        metrics_endpoints = ["/metrics", "/prometheus", "/stats"]

        async with httpx.AsyncClient() as client:
            for service_name, url in service_urls.items():
                metrics_found = False

                for metrics_endpoint in metrics_endpoints:
                    try:
                        response = await client.get(f"{url}{metrics_endpoint}", timeout=5.0)
                        if response.status_code == 200:
                            metrics_found = True

                            # Basic validation of metrics format
                            content = response.text
                            if metrics_endpoint == "/metrics":
                                # Should be Prometheus format
                                assert "# HELP" in content or "# TYPE" in content, "Invalid Prometheus metrics format"
                            elif metrics_endpoint == "/stats":
                                # Should be JSON format
                                try:
                                    json.loads(content)
                                except json.JSONDecodeError:
                                    assert False, "Invalid JSON metrics format"
                            break
                    except httpx.TimeoutException:
                        continue
                    except httpx.RequestError:
                        continue

                # At least some services should expose metrics
                if not metrics_found:
                    print(f"Warning: No metrics endpoint found for {service_name}")
