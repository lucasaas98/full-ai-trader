from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import numpy as np
import pandas as pd
import pytest


@dataclass
class LoadTestResult:
    """Result of a load test execution"""

    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    max_response_time: float
    min_response_time: float
    requests_per_second: float
    error_rate: float
    throughput_mbps: float
    start_time: datetime
    end_time: datetime
    duration: float


@dataclass
class PerformanceMetrics:
    """Performance metrics for system components"""

    cpu_usage_percent: float
    memory_usage_mb: float
    network_io_mbps: float
    disk_io_mbps: float
    database_connections: int
    redis_connections: int
    active_threads: int
    response_times: List[float]
    error_count: int


class LoadTestRunner:
    """Main load testing framework"""

    def __init__(self, base_url: str = "http://localhost", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[Dict[str, Any]] = []

    async def __aenter__(self) -> "LoadTestRunner":
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def make_request(
        self, method: str, endpoint: str, data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make a single HTTP request and measure performance"""
        start_time = time.time()
        error = None
        response_data = None
        status_code = None

        if self.session is None:
            raise RuntimeError("LoadTestRunner must be used as async context manager")

        try:
            url = f"{self.base_url}{endpoint}"

            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    status_code = response.status
                    response_data = await response.json()
            elif method.upper() == "POST":
                async with self.session.post(url, json=data) as response:
                    status_code = response.status
                    response_data = await response.json()
            elif method.upper() == "PUT":
                async with self.session.put(url, json=data) as response:
                    status_code = response.status
                    response_data = await response.json()
            elif method.upper() == "DELETE":
                async with self.session.delete(url) as response:
                    status_code = response.status
                    response_data = await response.json()

        except asyncio.TimeoutError:
            error = "timeout"
        except aiohttp.ClientError as e:
            error = f"client_error: {str(e)}"
        except Exception as e:
            error = f"unexpected_error: {str(e)}"

        end_time = time.time()
        response_time = end_time - start_time

        return {
            "response_time": response_time,
            "status_code": status_code,
            "success": error is None
            and status_code is not None
            and 200 <= status_code < 400,
            "error": error,
            "data": response_data,
            "timestamp": datetime.now(timezone.utc),
        }

    async def run_concurrent_requests(
        self,
        method: str,
        endpoint: str,
        num_requests: int,
        concurrency: int = 10,
        data_generator: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """Run concurrent requests and collect results"""

        semaphore = asyncio.Semaphore(concurrency)

        async def limited_request(request_id: int) -> Dict[str, Any]:
            async with semaphore:
                request_data = data_generator(request_id) if data_generator else None
                return await self.make_request(method, endpoint, request_data)

        tasks = [limited_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and convert to results
        valid_results: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                valid_results.append(
                    {
                        "response_time": self.timeout,
                        "status_code": None,
                        "success": False,
                        "error": str(result),
                        "data": None,
                        "timestamp": datetime.now(timezone.utc),
                    }
                )
            elif isinstance(result, dict):
                valid_results.append(result)

        return valid_results

    def analyze_results(
        self, results: List[Dict[str, Any]], test_name: str
    ) -> LoadTestResult:
        """Analyze load test results and generate report"""

        if not results:
            raise ValueError("No results to analyze")

        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]

        response_times = [r["response_time"] for r in results]
        successful_response_times = [r["response_time"] for r in successful_results]

        if not response_times:
            raise ValueError("No response time data available")

        start_time = min(r["timestamp"] for r in results)
        end_time = max(r["timestamp"] for r in results)
        duration = (end_time - start_time).total_seconds()

        # Calculate throughput (assuming average response size of 1KB)
        avg_response_size_bytes = 1024
        total_bytes = len(successful_results) * avg_response_size_bytes
        throughput_mbps = (
            (total_bytes / (1024 * 1024)) / duration if duration > 0 else 0
        )

        # Use successful response times for more accurate metrics
        metrics_response_times = (
            successful_response_times if successful_response_times else response_times
        )

        return LoadTestResult(
            test_name=test_name,
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            average_response_time=statistics.mean(metrics_response_times),
            median_response_time=statistics.median(metrics_response_times),
            p95_response_time=float(np.percentile(metrics_response_times, 95)),
            p99_response_time=float(np.percentile(metrics_response_times, 99)),
            max_response_time=max(metrics_response_times),
            min_response_time=min(metrics_response_times),
            requests_per_second=len(results) / duration if duration > 0 else 0,
            error_rate=len(failed_results) / len(results),
            throughput_mbps=throughput_mbps,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
        )


class TradingSystemLoadTests:
    """Load tests specific to trading system components"""

    @pytest.fixture
    async def load_test_runner(self) -> AsyncGenerator[LoadTestRunner, None]:
        """Create load test runner"""
        async with LoadTestRunner() as runner:
            yield runner

    @pytest.mark.performance
    async def test_data_collector_load(self, load_test_runner: LoadTestRunner) -> None:
        """Test data collector under load"""

        # Test health endpoint under load
        health_results = await load_test_runner.run_concurrent_requests(
            method="GET", endpoint=":9101/health", num_requests=1000, concurrency=50
        )

        health_analysis = load_test_runner.analyze_results(
            health_results, "data_collector_health"
        )

        # Assertions for health endpoint performance
        assert health_analysis.error_rate < 0.01  # Less than 1% error rate
        assert health_analysis.p95_response_time < 1.0  # 95% under 1 second
        assert health_analysis.requests_per_second > 100  # At least 100 RPS

        # Test market data endpoint under load
        def market_data_generator(request_id: int) -> Dict[str, str]:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            return {"symbol": symbols[request_id % len(symbols)]}

        market_data_results = await load_test_runner.run_concurrent_requests(
            method="POST",
            endpoint=":9101/api/market-data",
            num_requests=500,
            concurrency=25,
            data_generator=market_data_generator,
        )

        market_data_analysis = load_test_runner.analyze_results(
            market_data_results, "market_data_collection"
        )

        # Market data collection should handle moderate load
        assert market_data_analysis.error_rate < 0.05  # Less than 5% error rate
        assert market_data_analysis.p95_response_time < 2.0  # 95% under 2 seconds

    @pytest.mark.performance
    async def test_strategy_engine_load(self, load_test_runner: LoadTestRunner) -> None:
        """Test strategy engine under load"""

        def signal_generation_data(request_id: int) -> Dict[str, Any]:
            symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            return {
                "symbol": symbols[request_id % len(symbols)],
                "timeframe": "1m",
                "strategies": ["momentum", "mean_reversion"],
            }

        signal_results = await load_test_runner.run_concurrent_requests(
            method="POST",
            endpoint=":9102/api/generate-signals",
            num_requests=200,
            concurrency=20,
            data_generator=signal_generation_data,
        )

        signal_analysis = load_test_runner.analyze_results(
            signal_results, "signal_generation"
        )

        # Strategy execution should be efficient
        assert signal_analysis.error_rate < 0.02  # Less than 2% error rate
        assert signal_analysis.p95_response_time < 5.0  # 95% under 5 seconds
        assert signal_analysis.requests_per_second > 10  # At least 10 RPS

    @pytest.mark.performance
    async def test_risk_manager_load(self, load_test_runner: LoadTestRunner) -> None:
        """Test risk manager under load"""

        def risk_check_data(request_id: int) -> Dict[str, Any]:
            return {
                "signal": {
                    "signal_id": f"test_signal_{request_id}",
                    "symbol": "AAPL",
                    "signal_type": "BUY",
                    "strength": 0.8,
                    "price": 150.0 + (request_id % 10),
                    "quantity": 100,
                },
                "portfolio": {
                    "cash_balance": 50000.0,
                    "total_value": 100000.0,
                    "positions": {},
                },
            }

        risk_results = await load_test_runner.run_concurrent_requests(
            method="POST",
            endpoint=":9103/api/check-risk",
            num_requests=500,
            concurrency=30,
            data_generator=risk_check_data,
        )

        risk_analysis = load_test_runner.analyze_results(risk_results, "risk_checking")

        # Risk checks should be fast and reliable
        assert risk_analysis.error_rate < 0.01  # Less than 1% error rate
        assert risk_analysis.p95_response_time < 1.0  # 95% under 1 second
        assert risk_analysis.requests_per_second > 50  # At least 50 RPS

    @pytest.mark.performance
    async def test_trade_executor_load(self, load_test_runner: LoadTestRunner) -> None:
        """Test trade executor under load"""

        def order_data(request_id: int) -> Dict[str, Any]:
            return {
                "order": {
                    "symbol": "AAPL",
                    "side": "buy" if request_id % 2 == 0 else "sell",
                    "quantity": 100,
                    "order_type": "market",
                    "strategy_name": "test_strategy",
                }
            }

        execution_results = await load_test_runner.run_concurrent_requests(
            method="POST",
            endpoint=":9104/api/execute-order",
            num_requests=100,  # Fewer requests for trade execution
            concurrency=10,
            data_generator=order_data,
        )

        execution_analysis = load_test_runner.analyze_results(
            execution_results, "order_execution"
        )

        # Trade execution should be reliable but may be slower
        assert execution_analysis.error_rate < 0.05  # Less than 5% error rate
        assert execution_analysis.p95_response_time < 10.0  # 95% under 10 seconds

    @pytest.mark.performance
    async def test_portfolio_manager_load(
        self, load_test_runner: LoadTestRunner
    ) -> None:
        """Test scheduler service under load"""

        scheduler_results = await load_test_runner.run_concurrent_requests(
            method="GET", endpoint=":9105/health", num_requests=200, concurrency=20
        )

        scheduler_analysis = load_test_runner.analyze_results(
            scheduler_results, "scheduler_health"
        )

        # Scheduler should handle monitoring requests efficiently
        assert scheduler_analysis.error_rate < 0.01
        assert scheduler_analysis.p95_response_time < 0.5

    @pytest.mark.performance
    async def test_end_to_end_trading_flow_load(
        self, load_test_runner: LoadTestRunner
    ) -> None:
        """Test complete trading flow under load"""

        async def execute_trading_flow(
            session: aiohttp.ClientSession, flow_id: int
        ) -> Dict[str, Any]:
            """Execute a complete trading flow"""
            flow_start = time.time()

            try:
                # 1. Request market data
                market_data_start = time.time()
                async with session.post(
                    f"{load_test_runner.base_url}:9101/api/market-data",
                    json={"symbol": "AAPL"},
                ) as response:
                    market_data = await response.json()
                    market_data_time = time.time() - market_data_start
                    # Basic validation to ensure market_data is usable
                    if not market_data or not isinstance(market_data, dict):
                        raise ValueError("Invalid market data response")

                # 2. Generate trading signal
                signal_start = time.time()
                async with session.post(
                    f"{load_test_runner.base_url}:9102/api/generate-signals",
                    json={"symbol": "AAPL", "strategies": ["momentum"]},
                ) as response:
                    signals = await response.json()
                    signal_time = time.time() - signal_start

                # 3. Risk check
                risk_start = time.time()
                async with session.post(
                    f"{load_test_runner.base_url}:9103/api/check-risk",
                    json={
                        "signal": signals[0] if signals else {},
                        "portfolio": {"cash_balance": 50000},
                    },
                ) as response:
                    risk_result = await response.json()
                    risk_time = time.time() - risk_start

                # 4. Execute trade (if approved)
                execution_time = 0
                if risk_result.get("approved", False):
                    execution_start = time.time()
                    async with session.post(
                        f"{load_test_runner.base_url}:9104/api/execute-order",
                        json={
                            "order": {
                                "symbol": "AAPL",
                                "side": "buy",
                                "quantity": 100,
                                "order_type": "market",
                            }
                        },
                    ) as response:
                        await response.json()
                        execution_time = int(
                            (time.time() - execution_start) * 1000
                        )  # Convert to milliseconds

                total_time = int(
                    (time.time() - flow_start) * 1000
                )  # Convert to milliseconds

                return {
                    "flow_id": flow_id,
                    "success": True,
                    "total_time": total_time,
                    "market_data_time": market_data_time,
                    "signal_time": signal_time,
                    "risk_time": risk_time,
                    "execution_time": execution_time,
                    "timestamp": datetime.now(timezone.utc),
                }

            except Exception as e:
                return {
                    "flow_id": flow_id,
                    "success": False,
                    "error": str(e),
                    "total_time": time.time() - flow_start,
                    "timestamp": datetime.now(timezone.utc),
                }

        # Execute multiple trading flows concurrently
        semaphore = asyncio.Semaphore(10)  # Limit concurrency

        async def limited_flow(flow_id: int) -> Dict[str, Any]:
            async with semaphore:
                return await execute_trading_flow(load_test_runner.session, flow_id)

        # Run 50 complete trading flows
        flow_tasks = [limited_flow(i) for i in range(50)]
        flow_results = await asyncio.gather(*flow_tasks, return_exceptions=True)

        # Analyze end-to-end performance
        successful_flows = [
            r for r in flow_results if isinstance(r, dict) and r.get("success", False)
        ]

        assert len(successful_flows) > 40  # At least 80% success rate

        if successful_flows:
            avg_total_time = statistics.mean(r["total_time"] for r in successful_flows)
            assert (
                avg_total_time < 15.0
            )  # Complete flow should take less than 15 seconds

            # Component timing analysis
            avg_market_data_time = statistics.mean(
                r["market_data_time"] for r in successful_flows
            )
            avg_signal_time = statistics.mean(
                r["signal_time"] for r in successful_flows
            )
            avg_risk_time = statistics.mean(r["risk_time"] for r in successful_flows)

            assert avg_market_data_time < 2.0  # Market data should be fast
            assert avg_signal_time < 5.0  # Signal generation moderate time
            assert avg_risk_time < 1.0  # Risk checks should be fast


class DatabaseLoadTests:
    """Load tests for database operations"""

    @pytest.mark.performance
    async def test_database_connection_pool_load(
        self, mock_db_manager: AsyncMock
    ) -> None:
        """Test database connection pool under load"""

        async def db_operation(operation_id: int) -> Dict[str, Any]:
            """Simulate database operation"""
            start_time = time.time()

            try:
                # Simulate different types of database operations
                if operation_id % 4 == 0:
                    # Read operation
                    await mock_db_manager.fetch_all(
                        "SELECT * FROM market_data WHERE symbol = %s LIMIT 100",
                        ("AAPL",),
                    )
                elif operation_id % 4 == 1:
                    # Insert operation
                    await mock_db_manager.insert(
                        "market_data",
                        {
                            "symbol": "AAPL",
                            "price": 150.0,
                            "volume": 1000000,
                            "timestamp": datetime.now(timezone.utc),
                        },
                    )
                elif operation_id % 4 == 2:
                    # Update operation
                    await mock_db_manager.update(
                        "positions", {"unrealized_pnl": 500.0}, {"symbol": "AAPL"}
                    )
                else:
                    # Complex query
                    await mock_db_manager.fetch_all(
                        """
                        SELECT symbol, AVG(price) as avg_price, COUNT(*) as count
                        FROM market_data
                        WHERE timestamp > %s
                        GROUP BY symbol
                        """,
                        (datetime.now(timezone.utc) - timedelta(hours=1),),
                    )

                return {
                    "operation_id": operation_id,
                    "success": True,
                    "response_time": time.time() - start_time,
                }

            except Exception as e:
                return {
                    "operation_id": operation_id,
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time,
                }

        # Run 500 concurrent database operations
        semaphore = asyncio.Semaphore(20)  # Limit concurrent DB connections

        async def limited_db_operation(operation_id: int) -> Dict[str, Any]:
            async with semaphore:
                return await db_operation(operation_id)

        db_tasks = [limited_db_operation(i) for i in range(500)]
        db_results = await asyncio.gather(*db_tasks)

        # Analyze database performance
        successful_ops = [r for r in db_results if r["success"]]

        assert len(successful_ops) > 475  # At least 95% success rate

        if successful_ops:
            avg_response_time = statistics.mean(
                r["response_time"] for r in successful_ops
            )
            assert avg_response_time < 0.1  # Database operations should be fast

    @pytest.mark.performance
    async def test_redis_load(self, mock_redis_client: MagicMock) -> None:
        """Test Redis operations under load"""

        async def redis_operation(operation_id: int) -> Dict[str, Any]:
            """Simulate Redis operation"""
            start_time = time.time()

            try:
                key = f"test_key_{operation_id % 100}"  # Reuse some keys
                value = {"data": f"test_value_{operation_id}", "timestamp": time.time()}

                if operation_id % 3 == 0:
                    # Set operation
                    mock_redis_client.set(key, json.dumps(value))
                elif operation_id % 3 == 1:
                    # Get operation
                    mock_redis_client.get(key)
                else:
                    # Pub/sub operation
                    mock_redis_client.publish("test_channel", json.dumps(value))

                return {
                    "operation_id": operation_id,
                    "success": True,
                    "response_time": time.time() - start_time,
                }

            except Exception as e:
                return {
                    "operation_id": operation_id,
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time,
                }

        # Run 1000 Redis operations
        redis_tasks = [redis_operation(i) for i in range(1000)]
        redis_results = await asyncio.gather(*redis_tasks)

        successful_ops = [r for r in redis_results if r["success"]]

        # Redis should handle high load efficiently
        assert len(successful_ops) > 990  # Very high success rate expected

        if successful_ops:
            avg_response_time = statistics.mean(
                r["response_time"] for r in successful_ops
            )
            assert avg_response_time < 0.01  # Redis should be very fast

    @pytest.mark.performance
    async def test_message_queue_throughput(self, mock_redis_client: MagicMock) -> None:
        """Test Redis message queue throughput"""

        message_count = 1000
        publishers = 5
        subscribers = 3

        # Publisher coroutine
        async def publisher(
            publisher_id: int, messages_per_publisher: int
        ) -> Dict[str, Any]:
            published_count = 0
            start_time = time.time()

            for i in range(messages_per_publisher):
                message = {
                    "publisher_id": publisher_id,
                    "message_id": i,
                    "timestamp": time.time(),
                    "data": f"test_message_{publisher_id}_{i}",
                }

                mock_redis_client.publish("trading_signals", json.dumps(message))
                published_count += 1

                # Small delay to simulate realistic publishing rate
                await asyncio.sleep(0.001)

            return {
                "publisher_id": publisher_id,
                "published_count": published_count,
                "duration": time.time() - start_time,
            }

        # Subscriber coroutine
        async def subscriber(
            subscriber_id: int, expected_messages: int
        ) -> Dict[str, Any]:
            received_count = 0
            start_time = time.time()
            timeout_time = start_time + 30  # 30 second timeout

            while received_count < expected_messages and time.time() < timeout_time:
                # Simulate message reception
                mock_redis_client.get("trading_signals")
                received_count += 1
                await asyncio.sleep(0.001)

            return {
                "subscriber_id": subscriber_id,
                "received_count": received_count,
                "duration": time.time() - start_time,
            }

        messages_per_publisher = message_count // publishers

        # Start publishers and subscribers concurrently
        publisher_tasks = [
            publisher(i, messages_per_publisher) for i in range(publishers)
        ]
        subscriber_tasks = [subscriber(i, message_count) for i in range(subscribers)]

        all_tasks = publisher_tasks + subscriber_tasks
        results = await asyncio.gather(*all_tasks)

        publisher_results = results[:publishers]
        subscriber_results = results[publishers:]

        # Analyze throughput
        total_published = sum(r["published_count"] for r in publisher_results)
        total_received = sum(r["received_count"] for r in subscriber_results)

        assert total_published >= message_count * 0.95  # At least 95% published
        assert (
            total_received >= total_published * 0.9 * subscribers
        )  # Most messages received by all subscribers


class MemoryLeakTests:
    """Tests for memory leak detection"""

    @pytest.mark.performance
    async def test_long_running_data_collection_memory(self) -> None:
        """Test for memory leaks in long-running data collection"""
        import gc

        import psutil

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Simulate long-running data collection
        for cycle in range(100):
            # Simulate data collection cycle
            mock_data = []
            for i in range(1000):
                data_point = {
                    "symbol": f"STOCK{i % 10}",
                    "price": 100.0 + np.random.random(),
                    "volume": int(np.random.random() * 1000000),
                    "timestamp": datetime.now(timezone.utc),
                }
                mock_data.append(data_point)

            # Process data
            df = pd.DataFrame(mock_data)
            processed = df.groupby("symbol").agg({"price": "mean", "volume": "sum"})

            # Clear references
            del mock_data
            del df
            del processed

            # Force garbage collection every 10 cycles
            if cycle % 10 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory

                # Memory growth should be reasonable (less than 100MB after 100 cycles)
                if cycle == 99:
                    assert memory_growth < 100, f"Memory grew by {memory_growth:.2f}MB"

    @pytest.mark.performance
    async def test_redis_connection_pool_stability(
        self, mock_redis_client: MagicMock
    ) -> None:
        """Test Redis connection pool stability under load"""

        # Simulate rapid connection usage
        async def redis_stress_test(test_id: int) -> int:
            for i in range(100):
                key = f"stress_test_{test_id}_{i}"
                value = f"value_{i}" * 100  # Larger values

                # Rapid Redis operations
                mock_redis_client.set(key, value)
                mock_redis_client.get(key)
                mock_redis_client.delete(key)

                if i % 10 == 0:
                    await asyncio.sleep(0.001)  # Brief pause

            return test_id

        # Run multiple stress tests concurrently
        stress_tasks = [redis_stress_test(i) for i in range(20)]
        await asyncio.gather(*stress_tasks)

        # Connection pool should remain stable
        # This test mainly ensures no exceptions are raised

    @pytest.mark.performance
    async def test_websocket_connection_stability(self) -> None:
        """Test WebSocket connection stability under load"""

        class MockWebSocket:
            def __init__(self) -> None:
                self.connected = True
                self.message_count = 0

            async def send(self, message: str) -> None:
                if not self.connected:
                    raise Exception("Connection lost")
                self.message_count += 1

            async def recv(self) -> str:
                if not self.connected:
                    raise Exception("Connection lost")
                await asyncio.sleep(0.001)
                return json.dumps(
                    {
                        "symbol": "AAPL",
                        "price": 150.0 + np.random.random(),
                        "timestamp": time.time(),
                    }
                )

            async def close(self) -> None:
                self.connected = False

        # Simulate high-frequency WebSocket usage
        mock_ws = MockWebSocket()

        async def websocket_stress_test(duration_seconds: int = 30) -> Dict[str, float]:
            start_time = time.time()
            message_count = 0

            while time.time() - start_time < duration_seconds:
                try:
                    # Send heartbeat
                    await mock_ws.send(json.dumps({"type": "heartbeat"}))

                    # Receive data
                    await mock_ws.recv()
                    message_count += 1

                    # Brief processing delay
                    await asyncio.sleep(0.001)

                except Exception:
                    # Connection issues should be handled gracefully
                    break

            return {
                "duration": time.time() - start_time,
                "messages_processed": message_count,
                "messages_per_second": message_count / (time.time() - start_time),
            }

        result = await websocket_stress_test(30)

        # Should process many messages without issues
        assert result["messages_processed"] > 1000
        assert result["messages_per_second"] > 30


class LatencyOptimizationTests:
    """Tests for latency optimization"""

    @pytest.mark.performance
    async def test_signal_generation_latency(
        self, load_test_runner: LoadTestRunner
    ) -> None:
        """Test signal generation latency optimization"""

        # Test with different data sizes
        data_sizes = [100, 500, 1000, 2000]  # Number of historical data points
        latency_results = {}

        for data_size in data_sizes:

            def signal_data_generator(request_id: int) -> Dict[str, Any]:
                # Generate historical data of specified size
                historical_data = []
                for i in range(data_size):
                    historical_data.append(
                        {
                            "timestamp": (
                                datetime.now(timezone.utc) - timedelta(minutes=i)
                            ).isoformat(),
                            "open": 150.0 + np.random.random(),
                            "high": 151.0 + np.random.random(),
                            "low": 149.0 + np.random.random(),
                            "close": 150.0 + np.random.random(),
                            "volume": int(1000 + np.random.random() * 500),
                        }
                    )

                return {
                    "symbol": "AAPL",
                    "historical_data": historical_data,
                    "request_id": request_id,
                }

            async with load_test_runner:
                result = await load_test_runner.run_concurrent_requests(
                    "POST",
                    "/api/v1/signals/generate",
                    num_requests=50,
                    concurrency=10,
                    data_generator=signal_data_generator,
                )

            latency_results[data_size] = load_test_runner.analyze_results(
                f"signal_generation_data_size_{data_size}", result
            )

        # Verify latency increases with data size
        for i in range(1, len(data_sizes)):
            prev_size = data_sizes[i - 1]
            curr_size = data_sizes[i]

            assert (
                latency_results[curr_size].average_response_time
                >= latency_results[prev_size].average_response_time
            )
            assert (
                latency_results[curr_size].p95_response_time
                >= latency_results[prev_size].p95_response_time
            )
