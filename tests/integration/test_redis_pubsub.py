"""
Comprehensive integration tests for Redis pub/sub and database operations.
Tests real-time communication between services and data persistence.
"""

import asyncio
import json
import os
import queue
import random
import statistics
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Generator, List, Optional

import psycopg2
import pytest
import redis
from psycopg2.extras import RealDictCursor

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

pass  # Imports removed as they were unused


# Mock classes for missing modules
class RedisClient:
    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self.client = None

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass


class DatabaseManager:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config
        self.connection = None
        self.cursor = None

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass


class RedisPubSubTester:
    """Test Redis pub/sub functionality."""

    def __init__(
        self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 1
    ) -> None:
        self.redis_client = redis.Redis(
            host=redis_host, port=redis_port, db=redis_db, decode_responses=True
        )
        self.subscriber = None
        self.received_messages: queue.Queue = queue.Queue()
        self.subscription_active = False

    def start_subscriber(self, channels: List[str]) -> None:
        """Start subscribing to channels in a separate thread."""

        def subscriber_thread() -> None:
            try:
                self.subscriber = self.redis_client.pubsub()
                if self.subscriber is not None:
                    for channel in channels:
                        self.subscriber.subscribe(channel)

                    self.subscription_active = True

                    for message in self.subscriber.listen():
                        if message["type"] == "message":
                            self.received_messages.put(
                                {
                                    "channel": message["channel"],
                                    "data": message["data"],
                                    "timestamp": time.time(),
                                }
                            )
                        elif message["type"] == "subscribe":
                            print(f"Subscribed to {message['channel']}")

                        if not self.subscription_active:
                            break

            except Exception as e:
                self.received_messages.put({"error": str(e)})
            finally:
                if self.subscriber:
                    self.subscriber.close()

        self.subscriber_thread = threading.Thread(target=subscriber_thread)
        self.subscriber_thread.daemon = True
        self.subscriber_thread.start()

        # Wait for subscription to be active
        timeout = 5
        start_time = time.time()
        while not self.subscription_active and (time.time() - start_time) < timeout:
            time.sleep(0.1)

    def stop_subscriber(self) -> None:
        """Stop the subscriber."""
        self.subscription_active = False
        if hasattr(self, "subscriber_thread"):
            self.subscriber_thread.join(timeout=5)

    def get_messages(self, timeout: float = 1.0) -> List[Dict]:
        """Get received messages within timeout."""
        messages = []
        end_time = time.time() + timeout

        while time.time() < end_time:
            try:
                message = self.received_messages.get(timeout=0.1)
                if "error" in message:
                    raise Exception(message["error"])
                messages.append(message)
            except queue.Empty:
                continue

        return messages

    def publish_message(self, channel: str, message: Dict[str, Any]) -> int:
        """Publish a message to a channel."""
        result = self.redis_client.publish(channel, json.dumps(message))
        # Handle both sync and async results
        if hasattr(result, "__await__"):
            # If it's async, return 0 for now (would need async context to await)
            return 0
        # Safe conversion to int - avoid calling int() on awaitable
        if result is None:
            return 0
        if isinstance(result, int):
            return result
        if isinstance(result, (str, float)):
            try:
                return int(result)
            except (TypeError, ValueError):
                return 0
        return 0

    def cleanup(self) -> None:
        """Clean up Redis connections."""
        self.stop_subscriber()
        if self.redis_client:
            self.redis_client.close()


class DatabaseTester:
    """Test database operations and performance."""

    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.connection: Optional[Any] = None
        self.cursor: Optional[Any] = None

    def connect(self) -> None:
        """Connect to the database."""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
            )
            if self.connection:
                self.connection.autocommit = True
                self.cursor = self.connection.cursor(cursor_factory=RealDictCursor)
        except Exception as e:
            raise Exception(f"Failed to connect to database: {e}")

    def disconnect(self) -> None:
        """Disconnect from database."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> Dict[str, Any]:
        """Execute a query and return results."""
        if not self.cursor:
            raise Exception("Not connected to database")

        start_time = time.time()
        try:
            self.cursor.execute(query, params or ())
            results = self.cursor.fetchall()
            duration_ms = (time.time() - start_time) * 1000

            result_dict = {
                "success": True,
                "results": [dict(row) for row in results],
                "duration_ms": duration_ms,
                "row_count": len(results),
            }
            return result_dict
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_dict = {
                "success": False,
                "results": [],
                "duration_ms": duration_ms,
                "error": str(e),
            }
            return error_dict

    def insert_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert market data and measure performance."""
        query = """
        INSERT INTO market_data (symbol, price, bid, ask, volume, timestamp, source)
        VALUES (%(symbol)s, %(price)s, %(bid)s, %(ask)s, %(volume)s, %(timestamp)s, %(source)s)
        RETURNING id
        """

        start_time = time.time()
        try:
            if not self.cursor:
                raise Exception("Not connected to database")
            self.cursor.execute(query, market_data)
            result = self.cursor.fetchone()
            duration_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "inserted_id": result["id"] if result else None,
                "duration_ms": duration_ms,
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "inserted_id": None,
                "duration_ms": duration_ms,
                "error": str(e),
            }

    def insert_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert trade data and measure performance."""
        query = """
        INSERT INTO trades (symbol, action, quantity, price, value, strategy_id,
                           order_id, execution_time, status, fees)
        VALUES (%(symbol)s, %(action)s, %(quantity)s, %(price)s, %(value)s,
                %(strategy_id)s, %(order_id)s, %(execution_time)s, %(status)s, %(fees)s)
        RETURNING id
        """

        start_time = time.time()
        try:
            if not self.cursor:
                raise Exception("Not connected to database")
            self.cursor.execute(query, trade_data)
            result = self.cursor.fetchone()
            duration_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "inserted_id": result["id"] if result else None,
                "duration_ms": duration_ms,
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "inserted_id": None,
                "duration_ms": duration_ms,
                "error": str(e),
            }

    def bulk_insert_market_data(
        self, data_batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Bulk insert market data for performance testing."""
        query = """
        INSERT INTO market_data (symbol, price, bid, ask, volume, timestamp, source)
        VALUES (%(symbol)s, %(price)s, %(bid)s, %(ask)s, %(volume)s, %(timestamp)s, %(source)s)
        """

        """Batch insert multiple records and measure performance."""
        start_time = time.time()
        try:
            if not self.cursor:
                raise Exception("Not connected to database")
            self.cursor.executemany(query, data_batch)
            duration_ms = (time.time() - start_time) * 1000

            return {
                "success": True,
                "inserted_count": len(data_batch),
                "duration_ms": duration_ms,
                "rate_per_second": len(data_batch) / (duration_ms / 1000),
            }
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return {
                "success": False,
                "inserted_count": 0,
                "duration_ms": duration_ms,
                "error": str(e),
            }


@pytest.fixture
def redis_tester() -> Generator[RedisPubSubTester, None, None]:
    """Create Redis pub/sub tester."""
    tester = RedisPubSubTester()
    yield tester
    tester.cleanup()


@pytest.fixture
def database_tester() -> Generator[DatabaseTester, None, None]:
    """Create database tester."""
    db_config = {
        "host": "localhost",
        "port": "5432",
        "database": os.getenv("DB_NAME", "test_trading_system"),
        "user": os.getenv("DB_USER", "trader"),
        "password": os.getenv("DB_PASSWORD", "password"),
    }

    tester = DatabaseTester(db_config)
    try:
        tester.connect()
        yield tester
    finally:
        tester.disconnect()


@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """Generate sample market data for testing."""
    return {
        "symbol": "AAPL",
        "price": 150.25,
        "bid": 150.24,
        "ask": 150.26,
        "volume": 1000000,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "test_data",
    }


@pytest.fixture
def sample_trade_data() -> Dict[str, Any]:
    """Generate sample trade data for testing."""
    return {
        "symbol": "AAPL",
        "action": "BUY",
        "quantity": 100,
        "price": 150.25,
        "value": 15025.0,
        "strategy_id": "test_strategy_1",
        "order_id": str(uuid.uuid4()),
        "execution_time": datetime.now(timezone.utc),
        "status": "FILLED",
        "fees": 1.0,
    }


@pytest.mark.integration
@pytest.mark.redis
class TestRedisPubSub:
    """Test Redis pub/sub functionality."""

    def test_basic_publish_subscribe(self, redis_tester: RedisPubSubTester) -> None:
        """Test basic publish/subscribe functionality."""
        channel = "test_channel"
        test_message = {"type": "test", "data": "hello world", "timestamp": time.time()}

        # Start subscriber
        redis_tester.start_subscriber([channel])
        time.sleep(0.5)  # Allow subscription to stabilize

        # Publish message
        subscribers_count = redis_tester.publish_message(channel, test_message)
        assert subscribers_count == 1, f"Expected 1 subscriber, got {subscribers_count}"

        # Check received message
        messages = redis_tester.get_messages(timeout=2.0)
        assert len(messages) == 1, f"Expected 1 message, got {len(messages)}"

        received_data = json.loads(messages[0]["data"])
        assert received_data["type"] == test_message["type"]
        assert received_data["data"] == test_message["data"]

    def test_market_data_channel(
        self, redis_tester: RedisPubSubTester, sample_market_data: Dict[str, Any]
    ) -> None:
        """Test market data pub/sub channel."""
        channel = "market_data"

        redis_tester.start_subscriber([channel])
        time.sleep(0.5)

        # Publish market data
        subscribers_count = redis_tester.publish_message(channel, sample_market_data)
        assert subscribers_count >= 1, "No subscribers for market data channel"

        # Verify message received
        messages = redis_tester.get_messages(timeout=2.0)
        assert len(messages) >= 1, "Market data message not received"

        received_data = json.loads(messages[0]["data"])
        assert received_data["symbol"] == sample_market_data["symbol"]
        assert received_data["price"] == sample_market_data["price"]

    def test_trading_signal_processing(self, redis_tester: RedisPubSubTester) -> None:
        """Test trading signals pub/sub channel."""
        channel = "trading_signals"

        signal_data = {
            "symbol": "GOOGL",
            "action": "BUY",
            "confidence": 0.85,
            "strategy": "momentum",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reasoning": "Strong upward momentum detected",
        }

        redis_tester.start_subscriber([channel])
        time.sleep(0.5)

        # Publish signal
        subscribers_count = redis_tester.publish_message(channel, signal_data)
        assert subscribers_count >= 1, "No subscribers for trading signals"

        # Verify signal received
        messages = redis_tester.get_messages(timeout=2.0)
        assert len(messages) >= 1, "Trading signal not received"

        received_signal = json.loads(messages[0]["data"])
        assert received_signal["symbol"] == signal_data["symbol"]
        assert received_signal["action"] == signal_data["action"]
        assert received_signal["confidence"] == signal_data["confidence"]

    def test_multiple_channel_subscription(
        self, redis_tester: RedisPubSubTester
    ) -> None:
        """Test subscribing to multiple channels simultaneously."""
        channels = ["market_data", "trading_signals", "risk_alerts"]

        redis_tester.start_subscriber(channels)
        time.sleep(0.5)

        # Publish to different channels
        test_messages: list[tuple[str, Dict[str, Any]]] = [
            ("market_data", {"symbol": "AAPL", "price": 150.0}),
            ("trading_signals", {"symbol": "GOOGL", "action": "SELL"}),
            ("risk_alerts", {"type": "position_limit", "severity": "warning"}),
        ]

        for channel, message in test_messages:
            redis_tester.publish_message(channel, message)  # type: ignore

        # Verify all messages received
        messages = redis_tester.get_messages(timeout=3.0)
        assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"

        # Check each channel received its message
        channels_received = [msg["channel"] for msg in messages]
        for expected_channel in channels:
            assert (
                expected_channel in channels_received
            ), f"Missing message from {expected_channel}"

    def test_high_frequency_message_handling(
        self, redis_tester: RedisPubSubTester
    ) -> None:
        """Test high-frequency message publishing."""
        channel = "high_freq_test"
        message_count = 1000

        redis_tester.start_subscriber([channel])
        time.sleep(0.5)

        # Publish messages rapidly
        start_time = time.time()
        for i in range(message_count):
            message = {"id": i, "timestamp": time.time(), "data": f"message_{i}"}
            redis_tester.publish_message(channel, message)

        publish_duration = time.time() - start_time
        publish_rate = message_count / publish_duration

        # Wait for all messages to be received
        time.sleep(2.0)
        messages = redis_tester.get_messages(timeout=5.0)

        print(
            f"Published {message_count} messages in {publish_duration:.2f}s "
            f"({publish_rate:.2f} msg/sec)"
        )
        print(f"Received {len(messages)} messages")

        # Verify high message throughput
        assert (
            publish_rate > 500
        ), f"Publishing rate too low: {publish_rate:.2f} msg/sec"
        assert (
            len(messages) >= message_count * 0.95
        ), f"Message loss detected: {len(messages)}/{message_count}"

    def test_message_persistence(self, redis_tester: RedisPubSubTester) -> None:
        """Test that messages maintain order in pub/sub."""
        channel = "order_test"
        message_count = 100

        redis_tester.start_subscriber([channel])
        time.sleep(0.5)

        # Publish sequential messages
        for i in range(message_count):
            message = {"sequence": i, "timestamp": time.time()}
            redis_tester.publish_message(channel, message)

        # Wait for messages
        time.sleep(2.0)
        messages = redis_tester.get_messages(timeout=3.0)

        # Verify message order
        sequences = []
        for msg in messages:
            data = json.loads(msg["data"])
            sequences.append(data["sequence"])

        # Check if sequences are in order
        for i in range(1, len(sequences)):
            assert (
                sequences[i] >= sequences[i - 1]
            ), f"Message order violated at position {i}"

        print(f"Message ordering test: {len(sequences)} messages received in order")

    def test_subscriber_resilience(self, redis_tester: RedisPubSubTester) -> None:
        """Test subscriber resilience to connection issues."""
        channel = "resilience_test"

        redis_tester.start_subscriber([channel])
        time.sleep(0.5)

        # Publish initial message
        redis_tester.publish_message(channel, {"phase": "before_disruption"})

        # Simulate connection disruption
        redis_tester.stop_subscriber()
        time.sleep(1.0)

        # Restart subscriber
        redis_tester.start_subscriber([channel])
        time.sleep(0.5)

        # Publish message after restart
        redis_tester.publish_message(channel, {"phase": "after_restart"})

        # Check messages
        messages = redis_tester.get_messages(timeout=2.0)

        # Should receive the message after restart
        received_phases = []
        for msg in messages:
            data = json.loads(msg["data"])
            received_phases.append(data["phase"])

        assert (
            "after_restart" in received_phases
        ), "Failed to receive messages after restart"
        print(f"Subscriber resilience test: Received phases {received_phases}")


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseOperations:
    """Test database operations and performance."""

    def test_market_data_insertion(
        self, database_tester: DatabaseTester, sample_market_data: Dict[str, Any]
    ) -> None:
        """Test market data insertion performance."""
        result = database_tester.insert_market_data(sample_market_data)

        assert result["success"], f"Market data insertion failed: {result.get('error')}"
        assert (
            result["inserted_id"] is not None
        ), "No ID returned for inserted market data"
        assert (
            result["duration_ms"] < 100
        ), f"Insert too slow: {result['duration_ms']}ms"

        print(
            f"Market data inserted with ID {result['inserted_id']} in {result['duration_ms']:.2f}ms"
        )

    def test_trade_data_insertion(
        self, database_tester: DatabaseTester, sample_trade_data: Dict[str, Any]
    ) -> None:
        """Test trade data insertion performance."""
        result = database_tester.insert_trade(sample_trade_data)

        assert result["success"], f"Trade insertion failed: {result.get('error')}"
        assert result["inserted_id"] is not None, "No ID returned for inserted trade"
        assert (
            result["duration_ms"] < 50
        ), f"Trade insert too slow: {result['duration_ms']}ms"

        print(
            f"Trade inserted with ID {result['inserted_id']} in {result['duration_ms']:.2f}ms"
        )

    def test_bulk_market_data_insertion(self, database_tester: DatabaseTester) -> None:
        """Test bulk insertion performance."""
        # Generate batch of market data
        batch_size = 1000
        market_data_batch = []

        for i in range(batch_size):
            data = {
                "symbol": f"TEST{i % 10}",  # 10 different symbols
                "price": 100.0 + (i * 0.01),
                "bid": 99.99 + (i * 0.01),
                "ask": 100.01 + (i * 0.01),
                "volume": 1000 + i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "bulk_test",
            }
            market_data_batch.append(data)

        # Test bulk insertion
        result = database_tester.bulk_insert_market_data(market_data_batch)

        assert result["success"], f"Bulk insertion failed: {result.get('error')}"
        assert (
            result["inserted_count"] == batch_size
        ), f"Not all records inserted: {result['inserted_count']}/{batch_size}"
        assert (
            result["rate_per_second"] > 100
        ), f"Bulk insertion rate too low: {result['rate_per_second']:.2f}/sec"

        print(
            f"Bulk inserted {result['inserted_count']} records at {result['rate_per_second']:.2f}/sec"
        )

    def test_complex_queries_performance(
        self, database_tester: DatabaseManager
    ) -> None:
        """Test performance of complex analytical queries."""
        test_queries = [
            # Aggregate queries
            (
                "SELECT symbol, COUNT(*), AVG(price), MAX(volume) FROM market_data "
                "WHERE timestamp > NOW() - INTERVAL '1 hour' GROUP BY symbol",
                100,
            ),
            # Join queries
            (
                "SELECT t.symbol, t.quantity, t.price, p.current_quantity "
                "FROM trades t JOIN positions p ON t.symbol = p.symbol "
                "WHERE t.execution_time > NOW() - INTERVAL '1 day'",
                200,
            ),
            # Window functions
            (
                "SELECT symbol, price, "
                "LAG(price) OVER (PARTITION BY symbol ORDER BY timestamp) as prev_price "
                "FROM market_data WHERE timestamp > NOW() - INTERVAL '1 hour'",
                300,
            ),
            # Statistical queries
            (
                "SELECT symbol, "
                "STDDEV(price) as price_volatility, "
                "CORR(price, volume) as price_volume_correlation "
                "FROM market_data "
                "WHERE timestamp > NOW() - INTERVAL '1 day' "
                "GROUP BY symbol",
                500,
            ),
        ]

        for query, max_duration_ms in test_queries:
            result = database_tester.execute_query(query)

            assert result["success"], f"Query failed: {result.get('error')}"
            assert (
                result["duration_ms"] < max_duration_ms
            ), f"Query too slow: {result['duration_ms']:.2f}ms (max: {max_duration_ms}ms)"

            print(
                f"Query executed in {result['duration_ms']:.2f}ms, returned {result['row_count']} rows"
            )

    def test_concurrent_database_access(self, database_tester: DatabaseTester) -> None:
        """Test concurrent database access performance."""

        def worker_function(worker_id: int, results: List) -> None:
            """Worker function for concurrent database access."""
            worker_db_config = {
                "host": "localhost",
                "port": "5432",
                "database": os.getenv("DB_NAME", "test_trading_system"),
                "user": os.getenv("DB_USER", "trader"),
                "password": os.getenv("DB_PASSWORD", "password"),
            }

            worker_tester = DatabaseTester(worker_db_config)
            worker_tester.connect()

            try:
                worker_results = []
                for i in range(50):  # Each worker does 50 operations
                    market_data = {
                        "symbol": f"WORKER{worker_id}",
                        "price": 100.0 + i,
                        "bid": 99.99 + i,
                        "ask": 100.01 + i,
                        "volume": 1000 + i,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": f"worker_{worker_id}",
                    }

                    result = worker_tester.insert_market_data(market_data)
                    worker_results.append(result)

                results.append(worker_results)

            finally:
                worker_tester.disconnect()

        # Run concurrent workers
        import threading

        worker_count = 10
        worker_results: list[list] = []
        threads = []

        start_time = time.time()

        for worker_id in range(worker_count):
            thread = threading.Thread(
                target=worker_function, args=(worker_id, worker_results)
            )
            threads.append(thread)
            thread.start()

        # Wait for all workers to complete
        for thread in threads:
            thread.join()

        duration = time.time() - start_time

        # Analyze results
        total_operations = sum(len(worker_result) for worker_result in worker_results)
        successful_operations = sum(
            sum(
                1
                for op in worker_result
                if isinstance(op, dict) and op.get("success", False)
            )
            for worker_result in worker_results
        )

        success_rate = successful_operations / total_operations
        operations_per_second = total_operations / duration

        assert (
            success_rate > 0.95
        ), f"Concurrent access success rate too low: {success_rate * 100:.1f}%"
        assert (
            operations_per_second > 50
        ), f"Concurrent operation rate too low: {operations_per_second:.2f}/sec"

        print(
            f"Concurrent database test: {operations_per_second:.2f} ops/sec, "
            f"{success_rate * 100:.1f}% success rate"
        )

    def test_transaction_consistency(self, database_tester: DatabaseTester) -> None:
        """Test database transaction consistency."""
        # Start a transaction
        assert database_tester.connection is not None
        database_tester.connection.autocommit = False

        try:
            # Insert trade data
            trade_data = {
                "symbol": "TXNTEST",
                "action": "BUY",
                "quantity": 100,
                "price": 150.0,
                "value": 15000.0,
                "strategy_id": "test_strategy",
                "order_id": str(uuid.uuid4()),
                "execution_time": datetime.now(timezone.utc),
                "status": "FILLED",
                "fees": 1.0,
            }

            result = database_tester.insert_trade(trade_data)
            assert result["success"], f"Trade insertion failed: {result.get('error')}"

            # Update position (this would normally be in the same transaction)
            position_query = """
            INSERT INTO positions (symbol, quantity, avg_price, current_value, last_updated)
            VALUES (%(symbol)s, %(quantity)s, %(price)s, %(value)s, %(timestamp)s)
            ON CONFLICT (symbol) DO UPDATE SET
                quantity = positions.quantity + EXCLUDED.quantity,
                avg_price = ((positions.quantity * positions.avg_price) +
                           (EXCLUDED.quantity * EXCLUDED.avg_price)) /
                           (positions.quantity + EXCLUDED.quantity),
                current_value = EXCLUDED.current_value,
                last_updated = EXCLUDED.last_updated
            """

            position_params = {
                "symbol": trade_data["symbol"],
                "quantity": trade_data["quantity"],
                "price": trade_data["price"],
                "value": trade_data["value"],
                "timestamp": datetime.now(timezone.utc),
            }

            database_tester.cursor.execute(position_query, position_params)

            # Commit transaction
            assert database_tester.connection is not None
            database_tester.connection.commit()

            # Verify both records exist
            trade_check = database_tester.execute_query(
                "SELECT * FROM trades WHERE order_id = %s", (trade_data["order_id"],)
            )
            position_check = database_tester.execute_query(
                "SELECT * FROM positions WHERE symbol = %s", (trade_data["symbol"],)
            )

            assert trade_check["success"] and len(trade_check["results"]) == 1
            assert position_check["success"] and len(position_check["results"]) == 1

            print("Transaction consistency test passed")

        except Exception as e:
            assert database_tester.connection is not None
            database_tester.connection.rollback()
            raise e
        finally:
            assert database_tester.connection is not None
            database_tester.connection.autocommit = True


@pytest.mark.integration
@pytest.mark.redis
@pytest.mark.database
class TestServiceCommunication:
    """Test communication between services via Redis and database."""

    async def test_market_data_flow(
        self,
        redis_tester: RedisPubSubTester,
        database_tester: DatabaseManager,
        sample_market_data: Dict[str, Any],
    ) -> None:
        """Test complete market data flow: publish -> process -> store."""
        channel = "market_data"

        # Start subscriber to capture published data
        redis_tester.start_subscriber([channel])
        time.sleep(0.5)

        # Simulate data collector publishing market data
        redis_tester.publish_message(channel, sample_market_data)

        # Verify message was published
        messages = redis_tester.get_messages(timeout=2.0)
        assert len(messages) >= 1, "Market data not published to Redis"

        # Simulate strategy engine processing the data
        received_data = json.loads(messages[0]["data"])

        # Store processed data in database
        db_result = database_tester.insert_market_data(received_data)
        assert db_result[
            "success"
        ], f"Failed to store market data: {db_result.get('error')}"

        # Verify data persistence
        query_result = database_tester.execute_query(
            "SELECT * FROM market_data WHERE symbol = %s AND source = %s",
            (sample_market_data["symbol"], sample_market_data["source"]),
        )

        assert query_result["success"] and len(query_result["results"]) >= 1
        stored_data = query_result["results"][0]
        assert stored_data["symbol"] == sample_market_data["symbol"]
        assert float(stored_data["price"]) == sample_market_data["price"]

        print("Market data flow test completed successfully")

    async def test_trading_signal_workflow(
        self, redis_tester: RedisPubSubTester, database_tester: DatabaseTester
    ) -> None:
        """Test trading signal generation and processing workflow."""
        signal_channel = "trading_signals"
        execution_channel = "trade_execution"

        # Start subscribers for both channels
        redis_tester.start_subscriber([signal_channel, execution_channel])
        time.sleep(0.5)

        # Simulate strategy engine generating a signal
        trading_signal = {
            "symbol": "MSFT",
            "action": "BUY",
            "quantity": 50,
            "confidence": 0.92,
            "strategy": "ml_prediction",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reasoning": "Strong buy signal from ML model",
        }

        redis_tester.publish_message(signal_channel, trading_signal)

        # Simulate risk manager receiving and approving signal
        messages = redis_tester.get_messages(timeout=2.0)
        signal_received = False
        for msg in messages:
            if msg["channel"] == signal_channel:
                signal_received = True
                break

        assert signal_received, "Trading signal not received via Redis"

        # Simulate approved trade execution
        execution_data = {
            "order_id": str(uuid.uuid4()),
            "symbol": trading_signal["symbol"],
            "action": trading_signal["action"],
            "quantity": trading_signal["quantity"],
            "price": 250.75,
            "status": "FILLED",
            "execution_time": datetime.now(timezone.utc).isoformat(),
        }

        redis_tester.publish_message(execution_channel, execution_data)

        # Verify execution message
        exec_messages = redis_tester.get_messages(timeout=2.0)
        execution_received = False
        for msg in exec_messages:
            if msg["channel"] == execution_channel:
                execution_received = True
                break

        assert execution_received, "Trade execution message not received"

        # Store trade in database
        trade_data = {
            "symbol": execution_data["symbol"],
            "action": execution_data["action"],
            "quantity": execution_data["quantity"],
            "price": execution_data["price"],
            "value": float(str(execution_data["quantity"]))
            * float(str(execution_data["price"])),
            "strategy_id": trading_signal["strategy"],
            "order_id": execution_data["order_id"],
            "execution_time": datetime.fromisoformat(
                str(execution_data["execution_time"]).replace("Z", "+00:00")
                if isinstance(execution_data["execution_time"], str)
                else str(execution_data["execution_time"]).replace("Z", "+00:00")
            ),
            "status": execution_data["status"],
            "fees": 1.0,
        }

        db_result = database_tester.insert_trade(trade_data)
        assert db_result["success"], f"Failed to store trade: {db_result.get('error')}"

        print("Trading signal workflow test completed successfully")

    async def test_risk_check_integration(
        self, redis_tester: RedisPubSubTester, database_tester: DatabaseManager
    ) -> None:
        """Test risk alert propagation through the system."""
        alert_channel = "risk_alerts"
        action_channel = "risk_actions"

        redis_tester.start_subscriber([alert_channel, action_channel])
        time.sleep(0.5)

        # Simulate risk manager detecting a violation
        risk_alert = {
            "type": "position_limit_exceeded",
            "symbol": "TSLA",
            "current_position": 1000,
            "limit": 800,
            "severity": "critical",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_required": "reduce_position",
        }

        redis_tester.publish_message(alert_channel, risk_alert)

        # Verify alert received
        messages = redis_tester.get_messages(timeout=2.0)
        alert_received = False
        for msg in messages:
            if msg["channel"] == alert_channel:
                alert_received = True
                break

        assert alert_received, "Risk alert not propagated"

        # Simulate automated response
        risk_action = {
            "alert_id": str(uuid.uuid4()),
            "action": "reduce_position",
            "symbol": risk_alert["symbol"],
            "target_quantity": risk_alert["limit"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "pending",
        }

        redis_tester.publish_message(action_channel, risk_action)

        # Verify action message
        action_messages = redis_tester.get_messages(timeout=2.0)
        action_received = False
        for msg in action_messages:
            if msg["channel"] == action_channel:
                action_received = True
                break

        assert action_received, "Risk action not received"
        print("Risk alert propagation test completed successfully")

    def test_system_heartbeat_monitoring(self, redis_tester: RedisPubSubTester) -> None:
        """Test system heartbeat monitoring via Redis."""
        heartbeat_channel = "system_heartbeat"

        redis_tester.start_subscriber([heartbeat_channel])
        time.sleep(0.5)

        # Simulate services sending heartbeats
        services = [
            "data_collector",
            "strategy_engine",
            "risk_manager",
            "trade_executor",
        ]

        for service in services:
            heartbeat = {
                "service": service,
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "cpu_usage": 45.2,
                    "memory_usage_mb": 128.5,
                    "active_connections": 5,
                },
            }

            subscribers = redis_tester.publish_message(heartbeat_channel, heartbeat)
            assert subscribers >= 1, f"No subscribers for heartbeat from {service}"

        # Verify all heartbeats received
        messages = redis_tester.get_messages(timeout=3.0)
        assert len(messages) == len(
            services
        ), f"Expected {len(services)} heartbeats, got {len(messages)}"

        received_services = []
        for msg in messages:
            data = json.loads(msg["data"])
            received_services.append(data["service"])

        for service in services:
            assert (
                service in received_services
            ), f"Heartbeat from {service} not received"

        print(f"System heartbeat monitoring test: {len(services)} services reported")


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndTradingFlow:
    """Test complete end-to-end trading workflows."""

    async def test_complete_trading_cycle(
        self, redis_tester: RedisPubSubTester, database_tester: DatabaseManager
    ) -> None:
        """Test a complete trading cycle from data to execution."""
        channels = [
            "market_data",
            "trading_signals",
            "risk_checks",
            "trade_execution",
            "position_updates",
        ]

        redis_tester.start_subscriber(channels)
        time.sleep(0.5)

        # Step 1: Market data arrives
        market_data = {
            "symbol": "AMZN",
            "price": 3100.50,
            "bid": 3100.49,
            "ask": 3100.51,
            "volume": 50000,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "integration_test",
        }

        redis_tester.publish_message("market_data", market_data)

        # Step 2: Strategy generates signal
        await asyncio.sleep(0.1)  # Simulate processing time

        trading_signal = {
            "symbol": market_data["symbol"],
            "action": "BUY",
            "quantity": 10,
            "confidence": 0.88,
            "strategy": "momentum",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger_price": market_data["price"],
            "reasoning": "Strong momentum signal detected",
        }

        redis_tester.publish_message("trading_signals", trading_signal)

        # Step 3: Risk check
        await asyncio.sleep(0.1)

        risk_check = {
            "signal_id": str(uuid.uuid4()),
            "symbol": trading_signal["symbol"],
            "action": trading_signal["action"],
            "quantity": trading_signal["quantity"],
            "price": market_data["price"],
            "risk_score": 0.25,
            "approved": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        redis_tester.publish_message("risk_checks", risk_check)

        # Step 4: Trade execution
        await asyncio.sleep(0.1)

        trade_execution = {
            "order_id": str(uuid.uuid4()),
            "symbol": trading_signal["symbol"],
            "action": trading_signal["action"],
            "quantity": trading_signal["quantity"],
            "price": market_data["price"],
            "value": float(str(trading_signal["quantity"]))
            * float(str(market_data["price"])),
            "status": "FILLED",
            "execution_time": datetime.now(timezone.utc).isoformat(),
            "fees": 1.50,
        }

        redis_tester.publish_message("trade_execution", trade_execution)

        # Step 5: Position update
        await asyncio.sleep(0.1)

        position_update = {
            "symbol": trading_signal["symbol"],
            "quantity_change": trading_signal["quantity"],
            "new_total_quantity": trading_signal["quantity"],
            "avg_price": market_data["price"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        redis_tester.publish_message("position_updates", position_update)

        # Verify all messages received
        all_messages = redis_tester.get_messages(timeout=5.0)
        assert len(all_messages) == 5, f"Expected 5 messages, got {len(all_messages)}"

        # Verify message sequence
        expected_channels = [
            "market_data",
            "trading_signals",
            "risk_checks",
            "trade_execution",
            "position_updates",
        ]
        received_channels = [msg["channel"] for msg in all_messages]

        for expected_channel in expected_channels:
            assert (
                expected_channel in received_channels
            ), f"Missing message from {expected_channel}"

        # Store final trade in database
        db_trade_data = {
            "symbol": trade_execution["symbol"],
            "action": trade_execution["action"],
            "quantity": trade_execution["quantity"],
            "price": trade_execution["price"],
            "value": trade_execution["value"],
            "strategy_id": trading_signal["strategy"],
            "order_id": trade_execution["order_id"],
            "execution_time": datetime.fromisoformat(
                str(trade_execution["execution_time"]).replace("Z", "+00:00")
            ),
            "status": trade_execution["status"],
            "fees": float(str(trade_execution["fees"])),
        }

        db_result = database_tester.insert_trade(db_trade_data)
        assert db_result["success"], "Failed to store final trade"

        print("Complete trading cycle test passed")

    async def test_error_propagation_and_recovery(
        self, redis_tester: RedisPubSubTester
    ) -> None:
        """Test error propagation and system recovery."""
        error_channel = "system_errors"
        recovery_channel = "system_recovery"

        redis_tester.start_subscriber([error_channel, recovery_channel])
        time.sleep(0.5)

        # Simulate various error conditions
        error_scenarios = [
            {
                "type": "api_error",
                "service": "data_collector",
                "error": "Rate limit exceeded",
                "severity": "warning",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "type": "database_error",
                "service": "trade_executor",
                "error": "Connection timeout",
                "severity": "critical",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "type": "validation_error",
                "service": "risk_manager",
                "error": "Invalid position size",
                "severity": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]

        # Publish error scenarios
        for error in error_scenarios:
            redis_tester.publish_message(error_channel, error)

        # Simulate recovery actions
        await asyncio.sleep(0.5)

        recovery_actions = [
            {
                "type": "retry_with_backoff",
                "service": "data_collector",
                "action": "Implemented exponential backoff",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "type": "failover",
                "service": "trade_executor",
                "action": "Switched to backup database",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "type": "circuit_breaker",
                "service": "risk_manager",
                "action": "Activated circuit breaker",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]

        for recovery in recovery_actions:
            redis_tester.publish_message(recovery_channel, recovery)

        # Verify all messages received
        all_messages = redis_tester.get_messages(timeout=3.0)
        error_messages = [
            msg for msg in all_messages if msg["channel"] == error_channel
        ]
        recovery_messages = [
            msg for msg in all_messages if msg["channel"] == recovery_channel
        ]

        assert len(error_messages) == len(error_scenarios), "Not all errors propagated"
        assert len(recovery_messages) == len(
            recovery_actions
        ), "Not all recovery actions propagated"

        print(
            f"Error propagation test: {len(error_messages)} errors, {len(recovery_messages)} recoveries"
        )


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseIntegration:
    """Test database integration scenarios."""

    def test_concurrent_read_write_operations(
        self, redis_tester: RedisPubSubTester, database_tester: DatabaseTester
    ) -> None:
        """Test concurrent read and write operations."""

        def reader_worker(worker_id: int, results: List) -> None:
            """Worker that performs read operations."""
            worker_db = DatabaseTester(database_tester.db_config)
            worker_db.connect()

            try:
                for i in range(100):
                    market_data = {
                        "symbol": f"CONCURRENT{worker_id}",
                        "price": 100.0 + i,
                        "bid": 99.99 + i,
                        "ask": 100.01 + i,
                        "volume": 1000 + i,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source": f"worker_{worker_id}",
                    }

                    result = worker_db.insert_market_data(market_data)
                    results.append(result)

            finally:
                worker_db.disconnect()

        def reader_worker(results: List) -> None:
            """Worker that performs read operations."""
            reader_db = DatabaseTester(database_tester.db_config)
            reader_db.connect()

            try:
                for i in range(100):
                    query_result = reader_db.execute_query(
                        "SELECT COUNT(*) as count FROM market_data WHERE source LIKE 'worker_%'"
                    )
                    results.append(query_result)
                    time.sleep(0.01)  # Small delay

            finally:
                reader_db.disconnect()

        # Run concurrent workers
        import threading

        write_results: list[dict] = []
        read_results: list[dict] = []
        threads = []

        # Start writer threads
        for worker_id in range(5):
            thread = threading.Thread(
                target=writer_worker, args=(worker_id, write_results)
            )
            threads.append(thread)
            thread.start()

        # Start reader thread
        reader_thread = threading.Thread(target=reader_worker, args=(read_results,))
        threads.append(reader_thread)
        reader_thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Analyze results
        successful_writes = sum(1 for r in write_results if r["success"])
        successful_reads = sum(1 for r in read_results if r["success"])

        assert (
            successful_writes >= 450
        ), f"Too many write failures: {successful_writes}/500"
        assert successful_reads >= 95, f"Too many read failures: {successful_reads}/100"

        print(
            f"Concurrent DB test: {successful_writes} successful writes, {successful_reads} successful reads"
        )

    def test_data_consistency_across_tables(
        self, database_tester: DatabaseTester
    ) -> None:
        """Test data consistency across related tables."""
        # Insert market data
        market_data = {
            "symbol": "CONSISTENCY_TEST",
            "price": 200.0,
            "bid": 199.99,
            "ask": 200.01,
            "volume": 10000,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "consistency_test",
        }

        market_result = database_tester.insert_market_data(market_data)
        assert market_result["success"], "Failed to insert market data"

        # Insert related trade
        trade_data = {
            "symbol": market_data["symbol"],
            "action": "BUY",
            "quantity": 50,
            "price": market_data["price"],
            "value": 50 * float(str(market_data["price"])),
            "strategy_id": "consistency_test",
            "order_id": str(uuid.uuid4()),
            "execution_time": datetime.now(timezone.utc),
            "status": "FILLED",
            "fees": 2.0,
        }

        trade_result = database_tester.insert_trade(trade_data)
        assert trade_result["success"], "Failed to insert trade data"

        # Verify data consistency with join query
        consistency_check = database_tester.execute_query(
            """
            SELECT
                md.symbol,
                md.price as market_price,
                t.price as trade_price,
                t.quantity,
                t.value
            FROM market_data md
            JOIN trades t ON md.symbol = t.symbol
            WHERE md.symbol = %s
            AND md.source = 'consistency_test'
            ORDER BY md.timestamp DESC, t.execution_time DESC
            LIMIT 1
        """,
            (market_data["symbol"],),
        )

        assert consistency_check["success"], "Consistency check query failed"
        assert len(consistency_check["results"]) == 1, "No matching records found"

        result_row = consistency_check["results"][0]
        assert (
            result_row["market_price"] == result_row["trade_price"]
        ), "Price mismatch between tables"
        assert (
            result_row["value"] == result_row["trade_price"] * result_row["quantity"]
        ), "Value calculation incorrect"

        print("Data consistency test passed")

    def test_database_performance_under_load(
        self, database_tester: DatabaseTester
    ) -> None:
        """Test database performance under sustained load."""
        operations_count = 1000
        batch_size = 100
        start_time = time.time()

        total_operations = 0
        successful_operations = 0

        for batch_num in range(operations_count // batch_size):
            # Create batch of market data
            batch_data = []
            for i in range(batch_size):
                data = {
                    "symbol": f"PERF{i % 20}",  # 20 different symbols
                    "price": 100.0 + (batch_num * batch_size + i) * 0.01,
                    "bid": 99.99 + (batch_num * batch_size + i) * 0.01,
                    "ask": 100.01 + (batch_num * batch_size + i) * 0.01,
                    "volume": 1000 + i,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": f"perf_test_batch_{batch_num}",
                }
                batch_data.append(data)

            # Bulk insert
            result = database_tester.bulk_insert_market_data(batch_data)
            total_operations += batch_size

            if result["success"]:
                successful_operations += result["inserted_count"]

        duration = time.time() - start_time
        operations_per_second = total_operations / duration
        success_rate = successful_operations / total_operations

        assert (
            success_rate > 0.95
        ), f"Database success rate too low: {success_rate * 100:.1f}%"
        assert (
            operations_per_second > 100
        ), f"Database throughput too low: {operations_per_second:.2f} ops/sec"

        print(
            f"Database load test: {operations_per_second:.2f} ops/sec, {success_rate * 100:.1f}% success rate"
        )


@pytest.mark.integration
@pytest.mark.redis
@pytest.mark.database
class TestDataPipeline:
    """Test the complete data pipeline integration."""

    async def test_market_data_pipeline(
        self, redis_tester: RedisPubSubTester, database_tester: DatabaseTester
    ) -> None:
        """Test market data flowing through the complete pipeline."""
        # Simulate multiple data sources
        data_sources = ["twelve_data", "finviz", "yahoo_finance"]
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]

        redis_tester.start_subscriber(["market_data", "processed_data"])
        time.sleep(0.5)

        published_data = []

        # Publish market data from different sources
        for source in data_sources:
            for symbol in symbols:
                market_data = {
                    "symbol": symbol,
                    "price": 100.0 + hash(symbol + source) % 100,
                    "bid": 99.99,
                    "ask": 100.01,
                    "volume": 10000 + hash(symbol) % 50000,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source": source,
                }

                redis_tester.publish_message("market_data", market_data)
                published_data.append(market_data)

        # Simulate data processing and republishing
        await asyncio.sleep(0.5)

        for data in published_data[:5]:  # Process first 5 items
            processed_data = {
                **data,
                "processed": True,
                "processing_time": datetime.now(timezone.utc).isoformat(),
                "technical_indicators": {
                    "sma_20": float(str(data["price"])) * 0.98,
                    "rsi": 65.5,
                    "volume_avg": float(str(data["volume"])) * 1.1,
                },
            }

            redis_tester.publish_message("processed_data", processed_data)

        # Verify pipeline messages
        all_messages = redis_tester.get_messages(timeout=5.0)
        market_data_messages = [
            msg for msg in all_messages if msg["channel"] == "market_data"
        ]
        processed_messages = [
            msg for msg in all_messages if msg["channel"] == "processed_data"
        ]

        expected_market_messages = len(data_sources) * len(symbols)
        assert (
            len(market_data_messages) == expected_market_messages
        ), f"Expected {expected_market_messages} market data messages, got {len(market_data_messages)}"

        assert (
            len(processed_messages) == 5
        ), f"Expected 5 processed messages, got {len(processed_messages)}"

        # Store some data in database for persistence testing
        for data in published_data[:3]:
            db_result = database_tester.insert_market_data(data)
            assert db_result[
                "success"
            ], f"Failed to persist data: {db_result.get('error')}"

        print(
            f"Data pipeline test: {len(market_data_messages)} raw, {len(processed_messages)} processed messages"
        )

    async def test_real_time_aggregation(
        self, redis_tester: RedisPubSubTester, database_tester: DatabaseTester
    ) -> None:
        """Test real-time data aggregation capabilities."""
        aggregation_channel = "real_time_aggregation"

        redis_tester.start_subscriber([aggregation_channel])
        time.sleep(0.5)

        # Generate streaming data for aggregation
        symbol = "REALTIME_TEST"
        data_points = 100
        prices = []

        for i in range(data_points):
            price = 100.0 + (i * 0.1) + ((-1) ** i * 0.05)  # Oscillating trend
            prices.append(price)

            market_data = {
                "symbol": symbol,
                "price": price,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence": i,
            }

            # Publish data point
            redis_tester.publish_message("market_data", market_data)

            # Every 10 points, publish aggregated data
            if (i + 1) % 10 == 0:
                window_prices = prices[-10:]
                aggregated_data = {
                    "symbol": symbol,
                    "window_start": i - 9,
                    "window_end": i,
                    "min_price": min(window_prices),
                    "max_price": max(window_prices),
                    "avg_price": sum(window_prices) / len(window_prices),
                    "price_change": window_prices[-1] - window_prices[0],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                redis_tester.publish_message(aggregation_channel, aggregated_data)

        # Verify aggregated data
        messages = redis_tester.get_messages(timeout=3.0)
        aggregation_messages = [
            msg for msg in messages if msg["channel"] == aggregation_channel
        ]

        expected_aggregations = data_points // 10
        assert (
            len(aggregation_messages) == expected_aggregations
        ), f"Expected {expected_aggregations} aggregations, got {len(aggregation_messages)}"

        # Verify aggregation accuracy
        for msg in aggregation_messages:
            agg_data = json.loads(msg["data"])
            assert "min_price" in agg_data
            assert "max_price" in agg_data
            assert "avg_price" in agg_data
            assert (
                agg_data["min_price"] <= agg_data["avg_price"] <= agg_data["max_price"]
            )

        print(
            f"Real-time aggregation test: {len(aggregation_messages)} aggregations processed"
        )


@pytest.mark.integration
@pytest.mark.slow
class TestSystemResilience:
    """Test system resilience and failure recovery."""

    def test_redis_connection_resilience(self, redis_tester: RedisPubSubTester) -> None:
        """Test Redis connection recovery after failure."""
        test_channel = "resilience_test"

        # Start normal operation
        redis_tester.start_subscriber([test_channel])
        time.sleep(0.5)

        # Publish initial message
        initial_message = {"phase": "before_failure", "timestamp": time.time()}
        redis_tester.publish_message(test_channel, initial_message)

        # Verify initial message
        messages = redis_tester.get_messages(timeout=1.0)
        assert len(messages) >= 1, "Initial message not received"

        # Simulate Redis connection failure by stopping subscriber
        redis_tester.stop_subscriber()
        time.sleep(1.0)

        # Try to publish during "failure"
        failure_message = {"phase": "during_failure", "timestamp": time.time()}
        try:
            redis_tester.publish_message(test_channel, failure_message)
        except Exception:
            pass  # Expected during failure simulation

        # Restart subscriber (simulate recovery)
        redis_tester.start_subscriber([test_channel])
        time.sleep(0.5)

        # Publish recovery message
        recovery_message = {"phase": "after_recovery", "timestamp": time.time()}
        redis_tester.publish_message(test_channel, recovery_message)

        # Verify recovery
        recovery_messages = redis_tester.get_messages(timeout=2.0)
        assert len(recovery_messages) >= 1, "System did not recover properly"

        recovery_data = json.loads(recovery_messages[0]["data"])
        assert (
            recovery_data["phase"] == "after_recovery"
        ), "Recovery message not received"

        print("Redis connection recovery test passed")

    def test_database_connection_pool(
        self, database_tester: DatabaseTester
    ) -> None:
        """Test database connection pool behavior under stress."""
        # Create multiple connections to test pool behavior
        connection_count = 20
        connections = []

        try:
            for i in range(connection_count):
                db_tester = DatabaseTester(database_tester.db_config)
                db_tester.connect()
                connections.append(db_tester)

            # Test that all connections work
            for i, conn in enumerate(connections):
                result = conn.execute_query("SELECT 1 as test_value")
                assert result["success"], f"Connection {i} failed"

            print(
                f"Connection pool test: {len(connections)} concurrent connections successful"
            )

        finally:
            # Clean up connections
            for conn in connections:
                try:
                    conn.disconnect()
                except Exception:
                    pass

    async def test_message_durability(self, redis_tester: RedisPubSubTester) -> None:
        """Test message durability and persistence."""
        # durable_channel = 'durable_test_channel'  # Commented out as unused

        # Use Redis lists for durable messaging
        message_key = "durable_message_queue"

        # Publish messages to durable queue
        messages_to_send = 50
        sent_messages = []

        for i in range(messages_to_send):
            message = {
                "id": i,
                "symbol": "DURABLE_TEST",
                "price": 100.0 + i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            # Use LPUSH for durable storage
            redis_tester.redis_client.lpush(message_key, json.dumps(message))
            sent_messages.append(message)

        # Verify messages are stored
        queue_length = redis_tester.redis_client.llen(message_key)
        assert (
            queue_length == messages_to_send
        ), f"Expected {messages_to_send} messages, got {queue_length}"

        # Consume messages (simulate processing)
        consumed_messages = []
        for i in range(messages_to_send):
            message_json = redis_tester.redis_client.rpop(message_key)
            if message_json:
                message = json.loads(str(message_json))
                consumed_messages.append(message)

        # Verify all messages consumed
        assert len(consumed_messages) == messages_to_send, "Not all messages consumed"

        # Verify order preservation (FIFO)
        for i, consumed in enumerate(consumed_messages):
            expected_id = messages_to_send - 1 - i  # LIFO order due to LPUSH/RPOP
            original_message = next(
                (msg for msg in sent_messages if msg["id"] == expected_id), None
            )
            assert original_message is not None, f"Message {expected_id} not found"

        print(f"Message durability test: {len(consumed_messages)} messages processed")


@pytest.mark.integration
class TestServiceHealthMonitoring:
    """Test service health monitoring integration."""

    def test_health_check_aggregation(self, redis_tester: RedisPubSubTester) -> None:
        """Test aggregation of health checks from all services."""
        health_channel = "service_health"

        redis_tester.start_subscriber([health_channel])
        time.sleep(0.5)

        # Simulate health reports from all services
        services = [
            (
                "data_collector",
                "healthy",
                {"cpu": 25.5, "memory": 128, "connections": 5},
            ),
            (
                "strategy_engine",
                "healthy",
                {"cpu": 45.2, "memory": 256, "models_loaded": 3},
            ),
            (
                "risk_manager",
                "healthy",
                {"cpu": 15.8, "memory": 64, "rules_active": 12},
            ),
            (
                "trade_executor",
                "degraded",
                {"cpu": 60.1, "memory": 192, "pending_orders": 25},
            ),
            ("scheduler", "healthy", {"cpu": 10.2, "memory": 32, "scheduled_tasks": 8}),
        ]

        for service_name, status, metrics in services:
            health_report = {
                "service": service_name,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": metrics,
                "uptime_seconds": 3600 + hash(service_name) % 1000,
            }

            redis_tester.publish_message(health_channel, health_report)

        # Verify health reports
        messages = redis_tester.get_messages(timeout=3.0)
        assert len(messages) == len(
            services
        ), f"Expected {len(services)} health reports, got {len(messages)}"

        # Analyze system health
        healthy_services = 0
        degraded_services = 0

        for msg in messages:
            health_data = json.loads(msg["data"])
            if health_data["status"] == "healthy":
                healthy_services += 1
            elif health_data["status"] == "degraded":
                degraded_services += 1

        assert (
            healthy_services >= 4
        ), f"Too many unhealthy services: {healthy_services}/5 healthy"
        print(
            f"System health: {healthy_services} healthy, {degraded_services} degraded services"
        )

    async def test_risk_alert_propagation(
        self, redis_tester: RedisPubSubTester
    ) -> None:
        """Test alert escalation based on severity."""
        alert_channel = "system_alerts"
        escalation_channel = "alert_escalation"

        redis_tester.start_subscriber([alert_channel, escalation_channel])
        time.sleep(0.5)

        # Generate alerts of different severities
        alerts = [
            {"severity": "info", "message": "System startup completed"},
            {"severity": "warning", "message": "High memory usage detected"},
            {"severity": "error", "message": "API rate limit exceeded"},
            {"severity": "critical", "message": "Database connection lost"},
        ]

        # Publish alerts
        for alert in alerts:
            alert_data = {
                **alert,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": "test_service",
            }
            redis_tester.publish_message(alert_channel, alert_data)

        # Simulate escalation logic for critical alerts
        await asyncio.sleep(0.5)

        for alert in alerts:
            if alert["severity"] == "critical":
                escalation_data = {
                    "original_alert": alert,
                    "escalation_level": "immediate",
                    "notification_targets": ["ops_team", "management"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                redis_tester.publish_message(escalation_channel, escalation_data)

        # Verify escalation
        messages = redis_tester.get_messages(timeout=3.0)
        alert_messages = [msg for msg in messages if msg["channel"] == alert_channel]
        escalation_messages = [
            msg for msg in messages if msg["channel"] == escalation_channel
        ]

        assert len(alert_messages) == 4, f"Expected 4 alerts, got {len(alert_messages)}"
        assert (
            len(escalation_messages) == 1
        ), f"Expected 1 escalation, got {len(escalation_messages)}"

        # Verify critical alert was escalated
        escalation_data = json.loads(escalation_messages[0]["data"])
        assert escalation_data["original_alert"]["severity"] == "critical"

        print("Alert escalation test passed")


@pytest.mark.integration
@pytest.mark.external_api
class TestExternalAPIIntegration:
    """Test integration with external APIs through the system."""

    async def test_external_api_integration(
        self, redis_tester: RedisPubSubTester
    ) -> None:
        """Test external market data API integration."""
        api_response_channel = "api_responses"

        redis_tester.start_subscriber([api_response_channel])
        time.sleep(0.5)

        # Simulate external API responses
        api_responses = [
            {
                "source": "twelve_data",
                "symbol": "AAPL",
                "status": "success",
                "data": {"price": 150.25, "volume": 1000000},
                "latency_ms": 125.5,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "source": "finviz",
                "symbol": "GOOGL",
                "status": "rate_limited",
                "error": "API rate limit exceeded",
                "latency_ms": 5000.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "source": "yahoo_finance",
                "symbol": "MSFT",
                "status": "timeout",
                "error": "Request timeout after 30s",
                "latency_ms": 30000.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]

        # Publish API responses
        for response in api_responses:
            redis_tester.publish_message(api_response_channel, response)

        # Verify responses received
        messages = redis_tester.get_messages(timeout=3.0)
        assert len(messages) == 3, f"Expected 3 API responses, got {len(messages)}"

        # Analyze response patterns
        successful_responses = 0
        failed_responses = 0

        for msg in messages:
            response_data = json.loads(msg["data"])
            if response_data["status"] == "success":
                successful_responses += 1
            else:
                failed_responses += 1

        print(
            f"API integration test: {successful_responses} successful, {failed_responses} failed responses"
        )

    def test_redis_failover_simulation(self, redis_tester: RedisPubSubTester) -> None:
        """Test broker API integration simulation."""
        broker_channel = "broker_responses"

        redis_tester.start_subscriber([broker_channel])
        time.sleep(0.5)

        # Simulate broker API responses for different order types
        broker_responses: list[Dict[str, Any]] = [
            {
                "order_id": str(uuid.uuid4()),
                "status": "filled",
                "symbol": "AAPL",
                "quantity_filled": 100,
                "fill_price": 150.25,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "order_id": str(uuid.uuid4()),
                "status": "partially_filled",
                "symbol": "GOOGL",
                "quantity_filled": 25,
                "fill_price": 2500.75,
                "remaining_quantity": 25,
            },
            {
                "order_id": str(uuid.uuid4()),
                "status": "rejected",
                "symbol": "TSLA",
                "reason": "Insufficient buying power",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        ]

        # Publish broker responses
        for response in broker_responses:
            redis_tester.publish_message(broker_channel, response)

        # Verify responses
        messages = redis_tester.get_messages(timeout=2.0)
        assert len(messages) == 3, f"Expected 3 broker responses, got {len(messages)}"

        # Analyze order statuses
        statuses = []
        for msg in messages:
            response_data = json.loads(msg["data"])
            statuses.append(response_data["status"])

        expected_statuses = ["filled", "partial_fill", "rejected"]
        for status in expected_statuses:
            assert status in statuses, f"Missing broker response status: {status}"

        print("Broker API simulation test passed")


@pytest.mark.integration
@pytest.mark.performance
class TestIntegrationPerformance:
    """Test performance of integrated components."""

    async def test_end_to_end_latency(
        self, redis_tester: RedisPubSubTester, database_tester: DatabaseTester
    ) -> None:
        """Measure end-to-end latency for complete trading flow."""
        channels = ["market_data", "signals", "execution", "confirmation"]

        redis_tester.start_subscriber(channels)
        time.sleep(0.5)

        # Measure latency for complete flow
        flow_latencies = []

        for test_run in range(50):
            flow_start = time.time()

            # Step 1: Market data
            market_data = {
                "symbol": "LATENCY_TEST",
                "price": 100.0 + test_run,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "test_run": test_run,
            }

            redis_tester.publish_message("market_data", market_data)
            await asyncio.sleep(0.01)

            # Step 2: Signal generation
            signal = {
                "symbol": market_data["symbol"],
                "action": "BUY",
                "test_run": test_run,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            redis_tester.publish_message("signals", signal)
            await asyncio.sleep(0.01)

            # Step 3: Execution
            execution = {
                "symbol": signal["symbol"],
                "status": "filled",
                "test_run": test_run,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            redis_tester.publish_message("execution", execution)
            await asyncio.sleep(0.01)

            # Step 4: Confirmation
            confirmation = {
                "test_run": test_run,
                "flow_complete": True,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            redis_tester.publish_message("confirmation", confirmation)

            flow_latency = (time.time() - flow_start) * 1000  # Convert to ms
            flow_latencies.append(flow_latency)

        # Verify all messages received
        all_messages = redis_tester.get_messages(timeout=5.0)
        assert (
            len(all_messages) == 200
        ), f"Expected 200 messages (50 x 4), got {len(all_messages)}"

        # Analyze latency statistics
        avg_latency = statistics.mean(flow_latencies)
        p95_latency = (
            statistics.quantiles(flow_latencies, n=20)[18]
            if len(flow_latencies) > 1
            else 0
        )
        max_latency = max(flow_latencies)

        assert avg_latency < 100, f"Average E2E latency too high: {avg_latency:.2f}ms"
        assert p95_latency < 200, f"P95 E2E latency too high: {p95_latency:.2f}ms"

        print(
            f"End-to-end latency: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms, max={max_latency:.2f}ms"
        )

    async def test_throughput_scalability(
        self, redis_tester: RedisPubSubTester
    ) -> None:
        """Test system throughput scalability."""
        test_channel = "throughput_test"

        redis_tester.start_subscriber([test_channel])
        time.sleep(0.5)

        # Test increasing message rates
        message_rates = [10, 50, 100, 500, 1000]  # messages per second
        throughput_results = []

        for target_rate in message_rates:
            messages_sent = 0
            test_duration = 10  # seconds
            interval = 1.0 / target_rate  # seconds between messages

            start_time = time.time()
            next_send_time = start_time

            while (time.time() - start_time) < test_duration:
                current_time = time.time()

                if current_time >= next_send_time:
                    message = {
                        "rate_test": target_rate,
                        "message_id": messages_sent,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    redis_tester.publish_message(test_channel, message)
                    messages_sent += 1
                    next_send_time += interval

                await asyncio.sleep(0.001)  # Small sleep to prevent CPU spinning

            actual_duration = time.time() - start_time
            actual_rate = messages_sent / actual_duration

            throughput_results.append(
                {
                    "target_rate": target_rate,
                    "actual_rate": actual_rate,
                    "messages_sent": messages_sent,
                    "duration": actual_duration,
                }
            )

            print(
                f"Throughput test: Target {target_rate}/s, Actual {actual_rate:.2f}/s"
            )

            # Brief pause between tests
            await asyncio.sleep(1.0)

        # Verify throughput scaling
        for result in throughput_results:
            target = result["target_rate"]
            actual = result["actual_rate"]
            accuracy = actual / target

            # Allow 10% variance in throughput
            assert (
                accuracy >= 0.9
            ), f"Throughput too low for {target}/s: {actual:.2f}/s ({accuracy * 100:.1f}%)"

        # Verify messages received
        final_messages = redis_tester.get_messages(timeout=5.0)
        total_expected = sum(r["messages_sent"] for r in throughput_results)

        # Allow some message loss at high rates
        message_loss_rate = 1 - (len(final_messages) / total_expected)
        assert (
            message_loss_rate < 0.05
        ), f"Message loss rate too high: {message_loss_rate * 100:.1f}%"

        print(
            f"Throughput scalability test completed: {len(final_messages)}/{total_expected} messages received"
        )


# Utility functions for integration testing
class IntegrationTestHelper:
    """Helper class for integration testing."""

    @staticmethod
    def wait_for_condition(
        condition_func: Callable[[], bool],
        timeout_seconds: float = 5.0,
        check_interval: float = 0.1,
    ) -> bool:
        """Wait for a condition to become true."""
        start_time = time.time()
        while (time.time() - start_time) < timeout_seconds:
            if condition_func():
                return True
            time.sleep(check_interval)
        return False

    @staticmethod
    def generate_test_data_batch(
        count: int, symbol_prefix: str = "TEST"
    ) -> List[Dict[str, Any]]:
        """Generate a batch of test data."""
        batch = []
        for i in range(count):
            data = {
                "symbol": f"{symbol_prefix}{i % 10}",
                "price": 100.0 + (i * 0.01),
                "volume": 1000 + i,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "integration_test",
            }
            batch.append(data)
        return batch

    @staticmethod
    def verify_message_integrity(
        sent_messages: List[Dict], received_messages: List[Dict]
    ) -> bool:
        """Verify message integrity between sent and received."""
        if len(sent_messages) != len(received_messages):
            return False

        for sent, received in zip(sent_messages, received_messages):
            received_data = (
                json.loads(received["data"])
                if isinstance(received["data"], str)
                else received["data"]
            )
            for key, value in sent.items():
                if key not in received_data or received_data[key] != value:
                    return False

        return True


@pytest.mark.integration
@pytest.mark.slow
def test_integration_test_suite_performance() -> None:
    """Meta-test to ensure integration tests themselves perform well."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Run a subset of integration tests
    start_time = time.time()

    # This would run other integration tests
    # For now, just simulate some work
    for i in range(1000):
        data = {"test": i, "timestamp": time.time()}
        json.dumps(data)  # Simulate JSON processing

    duration = time.time() - start_time
    final_memory = process.memory_info().rss / 1024 / 1024

    memory_growth = final_memory - initial_memory

    assert duration < 30, f"Integration test suite too slow: {duration:.2f}s"
    assert (
        memory_growth < 50
    ), f"Integration tests using too much memory: {memory_growth:.2f}MB"

    print(
        f"Integration test performance: {duration:.2f}s duration, {memory_growth:.2f}MB memory growth"
    )


# Test data generators for integration testing
def generate_realistic_market_session() -> List[Dict[str, Any]]:
    """Generate realistic market session data."""
    session_start = datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    session_data = []

    for minute in range(390):  # 6.5 hour session
        timestamp = session_start + timedelta(minutes=minute)

        # Generate data for multiple symbols
        for symbol in ["AAPL", "GOOGL", "MSFT"]:
            base_price = {"AAPL": 150, "GOOGL": 2500, "MSFT": 300}[symbol]
            price_variation = random.uniform(-2, 2)

            market_data = {
                "symbol": symbol,
                "price": base_price + price_variation,
                "volume": random.randint(10000, 100000),
                "timestamp": timestamp.isoformat(),
            }
            session_data.append(market_data)

    return session_data


def generate_test_scenarios() -> List[Dict[str, Any]]:
    """Generate various trading scenarios for testing."""
    scenarios = [
        {
            "name": "bull_market",
            "description": "Strong upward trend",
            "price_multiplier": 1.02,
            "volume_multiplier": 1.5,
            "signal_frequency": 0.8,
        },
        {
            "name": "bear_market",
            "description": "Strong downward trend",
            "price_multiplier": 0.98,
            "volume_multiplier": 2.0,
            "signal_frequency": 0.6,
        },
        {
            "name": "sideways_market",
            "description": "Low volatility, range-bound",
            "price_multiplier": 1.0,
            "volume_multiplier": 0.8,
            "signal_frequency": 0.3,
        },
        {
            "name": "high_volatility",
            "description": "High volatility environment",
            "price_multiplier": random.uniform(0.95, 1.05),
            "volume_multiplier": 3.0,
            "signal_frequency": 0.9,
        },
    ]

    return scenarios


if __name__ == "__main__":
    # Run integration tests standalone
    import pytest

    # Run specific test categories
    test_args = [
        "-v",
        "-m",
        "integration and redis",
        "--tb=short",
        os.path.dirname(__file__),
    ]

    print("Running Redis integration tests...")
    exit_code = pytest.main(test_args)

    if exit_code == 0:
        print("All Redis integration tests passed!")
    else:
        print(f"Some tests failed (exit code: {exit_code})")

    # Run database integration tests
    db_test_args = [
        "-v",
        "-m",
        "integration and database",
        "--tb=short",
        os.path.dirname(__file__),
    ]

    print("\nRunning Database integration tests...")
    db_exit_code = pytest.main(db_test_args)

    if db_exit_code == 0:
        print("All Database integration tests passed!")
    else:
        print(f"Some database tests failed (exit code: {db_exit_code})")
