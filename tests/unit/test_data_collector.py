import asyncio
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from shared.models import AssetType, FinVizData, MarketData, TimeFrame

sys.path.append("/app/shared")


class TestMarketDataValidation:
    """Unit tests for market data validation business rules"""

    @pytest.mark.unit
    def test_market_data_model_validation(self):
        """Test MarketData model validation rules"""
        # Valid market data
        valid_data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(timezone.utc),
            timeframe=TimeFrame.ONE_MINUTE,
            open=Decimal("150.0"),
            high=Decimal("152.0"),
            low=Decimal("149.0"),
            close=Decimal("151.0"),
            volume=1000000,
            adjusted_close=Decimal("151.0"),
            asset_type=AssetType.STOCK,
        )

        assert valid_data.symbol == "AAPL"
        assert valid_data.high >= valid_data.open
        assert valid_data.high >= valid_data.close
        assert valid_data.low <= valid_data.open
        assert valid_data.low <= valid_data.close
        assert valid_data.volume > 0

    @pytest.mark.unit
    def test_market_data_price_validation_rules(self):
        """Test that price validation rules are enforced"""
        # High price must be >= other prices
        with pytest.raises(ValueError, match="high must be"):
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal("150.0"),
                high=Decimal("149.0"),  # Invalid: high < open
                low=Decimal("148.0"),
                close=Decimal("149.5"),
                volume=1000000,
                adjusted_close=Decimal("149.5"),
            )

    @pytest.mark.unit
    def test_market_data_negative_values_rejected(self):
        """Test that negative prices and volumes are rejected"""
        with pytest.raises(ValueError):
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal("-150.0"),  # Invalid: negative price
                high=Decimal("152.0"),
                low=Decimal("149.0"),
                close=Decimal("151.0"),
                volume=1000000,
                adjusted_close=Decimal("151.0"),
            )

        with pytest.raises(ValueError):
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(timezone.utc),
                timeframe=TimeFrame.ONE_MINUTE,
                open=Decimal("150.0"),
                high=Decimal("152.0"),
                low=Decimal("149.0"),
                close=Decimal("151.0"),
                volume=-1000000,  # Invalid: negative volume
                adjusted_close=Decimal("151.0"),
            )

    @pytest.mark.unit
    def test_timestamp_validation(self):
        """Test timestamp validation for market data"""
        # Future timestamps should be handled appropriately
        future_time = datetime.now(timezone.utc) + timedelta(hours=1)

        # This should work but business logic should handle future timestamps
        market_data = MarketData(
            symbol="AAPL",
            timestamp=future_time,
            timeframe=TimeFrame.ONE_MINUTE,
            open=Decimal("150.0"),
            high=Decimal("152.0"),
            low=Decimal("149.0"),
            close=Decimal("151.0"),
            volume=1000000,
            adjusted_close=Decimal("151.0"),
        )

        assert market_data.timestamp == future_time


class TestFinVizDataValidation:
    """Unit tests for FinViz data validation"""

    @pytest.mark.unit
    def test_finviz_data_creation(self):
        """Test FinViz data model creation and validation"""
        finviz_data = FinVizData(
            ticker="AAPL",
            symbol="AAPL",
            company="Apple Inc.",
            industry="Technology",
            country="USA",
            price=Decimal("150.25"),
            change=0.25,
            volume=1500000,
            market_cap=Decimal("2500000000000"),
            pe_ratio=25.5,
            sector="Technology",
        )

        assert finviz_data.ticker == "AAPL"
        assert finviz_data.price is not None and finviz_data.price > Decimal("0")
        assert finviz_data.volume is not None and finviz_data.volume > 0
        assert finviz_data.market_cap is not None and finviz_data.market_cap > Decimal(
            "0"
        )

    @pytest.mark.unit
    def test_finviz_data_negative_price_rejected(self):
        """Test that negative prices are rejected in FinViz data"""
        with pytest.raises(ValueError):
            FinVizData(
                ticker="AAPL",
                symbol="AAPL",
                company="Apple Inc.",
                industry="Technology",
                country="USA",
                price=Decimal("-150.25"),  # Invalid: negative price
                change=0.25,
                volume=1500000,
                market_cap=Decimal("2500000000000"),
                pe_ratio=25.5,
                sector="Technology",
            )


class TestDataQualityChecks:
    """Unit tests for data quality validation business logic"""

    @pytest.mark.unit
    def test_ohlcv_data_consistency(self):
        """Test OHLCV data consistency rules"""
        # Test data that violates OHLCV rules
        inconsistent_data = {
            "open": 100.0,
            "high": 99.0,  # High should be >= open
            "low": 101.0,  # Low should be <= open
            "close": 102.0,
            "volume": 1000,
        }

        # Business logic should detect this inconsistency
        def validate_ohlcv_consistency(data):
            high = data["high"]
            low = data["low"]
            open_price = data["open"]
            close = data["close"]

            # High should be >= all other prices
            if high < max(open_price, low, close):
                return False

            # Low should be <= all other prices
            if low > min(open_price, high, close):
                return False

            return True

        assert not validate_ohlcv_consistency(inconsistent_data)

        # Test consistent data
        consistent_data = {
            "open": 100.0,
            "high": 102.0,  # Properly highest
            "low": 99.0,  # Properly lowest
            "close": 101.0,
            "volume": 1000,
        }

        assert validate_ohlcv_consistency(consistent_data)

    @pytest.mark.unit
    def test_volume_anomaly_detection(self):
        """Test volume anomaly detection logic"""

        def detect_volume_anomaly(volumes, threshold_multiplier=3):
            """Detect volume anomalies using median-based analysis"""
            if len(volumes) < 2:
                return False

            # Use median instead of mean to avoid outlier contamination
            sorted_volumes = sorted(volumes)
            n = len(sorted_volumes)
            if n % 2 == 0:
                median = (sorted_volumes[n // 2 - 1] + sorted_volumes[n // 2]) / 2
            else:
                median = sorted_volumes[n // 2]

            for volume in volumes:
                if volume > threshold_multiplier * median:
                    return True
            return False

        # Normal volume data
        normal_volumes = [1000000, 1100000, 950000, 1200000, 1050000]
        assert not detect_volume_anomaly(normal_volumes)

        # Volume with anomaly - 10M is much higher than the median of ~1.1M
        anomaly_volumes = [
            1000000,
            1100000,
            10000000,
            1200000,
            1050000,
        ]  # 10M is anomaly
        assert detect_volume_anomaly(anomaly_volumes)

    @pytest.mark.unit
    def test_price_gap_detection(self):
        """Test price gap detection logic"""

        def detect_price_gaps(closes, threshold=0.05):  # 5% threshold
            """Detect significant price gaps between consecutive periods"""
            if len(closes) < 2:
                return False

            for i in range(1, len(closes)):
                if closes[i - 1] == 0:  # Avoid division by zero
                    continue
                change = abs(closes[i] - closes[i - 1]) / closes[i - 1]
                if change > threshold:
                    return True
            return False

        # Normal price progression
        normal_closes = [100.0, 100.5, 101.0, 100.8, 101.2]
        assert not detect_price_gaps(normal_closes)

        # Price with gap
        gap_closes = [100.0, 100.5, 110.0, 110.2, 109.8]  # 9.5% gap
        assert detect_price_gaps(gap_closes)

    @pytest.mark.unit
    def test_data_completeness_check(self):
        """Test data completeness validation"""

        def check_data_completeness(data_points, expected_count):
            """Check if we have the expected amount of data"""
            missing_count = expected_count - len(data_points)
            completeness_ratio = (
                len(data_points) / expected_count if expected_count > 0 else 0
            )

            return {
                "is_complete": missing_count == 0,
                "missing_count": missing_count,
                "completeness_ratio": completeness_ratio,
            }

        # Complete data
        complete_data = list(range(100))  # 100 data points
        result = check_data_completeness(complete_data, 100)
        assert result["is_complete"] is True
        assert result["completeness_ratio"] == 1.0

        # Incomplete data
        incomplete_data = list(range(80))  # Only 80 out of 100
        result = check_data_completeness(incomplete_data, 100)
        assert result["is_complete"] is False
        assert result["missing_count"] == 20
        assert result["completeness_ratio"] == 0.8


class TestDataProcessingLogic:
    """Unit tests for data processing business logic"""

    @pytest.mark.unit
    def test_timeframe_aggregation_logic(self):
        """Test timeframe aggregation business rules"""

        def aggregate_to_5min(minute_data):
            """Aggregate 1-minute bars to 5-minute bars"""
            if not minute_data:
                return None

            return {
                "timestamp": minute_data[0]["timestamp"],
                "open": minute_data[0]["open"],
                "high": max(d["high"] for d in minute_data),
                "low": min(d["low"] for d in minute_data),
                "close": minute_data[-1]["close"],
                "volume": sum(d["volume"] for d in minute_data),
            }

        # Create 5 minutes of 1-minute data
        base_time = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)
        minute_data = []

        for i in range(5):
            minute_data.append(
                {
                    "timestamp": base_time + timedelta(minutes=i),
                    "open": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "close": 100.5 + i,
                    "volume": 1000 + i * 100,
                }
            )

        five_min_bar = aggregate_to_5min(minute_data)

        assert five_min_bar is not None
        assert five_min_bar["open"] == 100.0  # First open
        assert five_min_bar["close"] == 104.5  # Last close
        assert five_min_bar["high"] == 105.0  # Highest high
        assert five_min_bar["low"] == 99.0  # Lowest low
        assert five_min_bar["volume"] == 6000  # Sum of volumes

    @pytest.mark.unit
    def test_data_cleaning_logic(self):
        """Test data cleaning business logic"""

        def clean_market_data(data):
            """Clean market data by removing invalid records"""
            cleaned = []
            for record in data:
                # Skip records with missing price
                if record.get("price") is None:
                    continue

                # Skip records with negative volume
                if record.get("volume", 0) < 0:
                    continue

                # Skip extreme price outliers (simple rule: > 10000)
                if record.get("price", 0) > 10000:
                    continue

                # Skip records with zero or negative prices
                if record.get("price", 0) <= 0:
                    continue

                cleaned.append(record)

            return cleaned

        dirty_data = [
            {"price": 100.0, "volume": 1000},
            {"price": None, "volume": 1100},  # Missing price
            {"price": 101.0, "volume": -500},  # Negative volume
            {"price": 999999.0, "volume": 1200},  # Outlier price
            {"price": 102.0, "volume": 1300},
            {"price": 0, "volume": 1400},  # Zero price
            {"price": -50.0, "volume": 1500},  # Negative price
        ]

        clean_result = clean_market_data(dirty_data)

        assert len(clean_result) == 2  # Only 2 valid records
        assert all(r["price"] is not None for r in clean_result)
        assert all(r["volume"] >= 0 for r in clean_result)
        assert all(r["price"] <= 10000 for r in clean_result)
        assert all(r["price"] > 0 for r in clean_result)

    @pytest.mark.unit
    def test_moving_average_calculation(self):
        """Test moving average calculation logic"""

        def calculate_sma(prices, window):
            """Calculate Simple Moving Average"""
            if len(prices) < window:
                return []

            sma_values = []
            for i in range(window - 1, len(prices)):
                window_prices = prices[i - window + 1 : i + 1]
                sma = sum(window_prices) / window
                sma_values.append(sma)

            return sma_values

        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        sma_5 = calculate_sma(prices, 5)

        # Should have 6 SMA values (10 prices - 5 window + 1)
        assert len(sma_5) == 6

        # First SMA should be average of first 5 prices
        expected_first_sma = sum(prices[:5]) / 5  # (100+101+102+103+104)/5 = 102
        assert abs(sma_5[0] - expected_first_sma) < 0.01


class TestErrorHandlingLogic:
    """Unit tests for error handling business logic"""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test retry logic for failed operations"""
        attempt_count = 0

        async def failing_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"

        async def retry_with_backoff(operation, max_retries=3, delay=0.01):
            """Retry an operation with exponential backoff"""
            for attempt in range(max_retries):
                try:
                    return await operation()
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    await asyncio.sleep(delay * (2**attempt))  # Exponential backoff

        result = await retry_with_backoff(failing_operation)
        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling logic"""

        async def slow_operation():
            await asyncio.sleep(0.1)  # 100ms operation
            return "completed"

        # Test successful operation within timeout
        result = await asyncio.wait_for(slow_operation(), timeout=0.2)
        assert result == "completed"

        # Test timeout scenario
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_operation(), timeout=0.05)

    @pytest.mark.unit
    def test_circuit_breaker_logic(self):
        """Test circuit breaker pattern for fault tolerance"""

        class CircuitBreaker:
            def __init__(self, failure_threshold=3, timeout=5):
                self.failure_threshold = failure_threshold
                self.timeout = timeout
                self.failure_count = 0
                self.last_failure_time = None
                self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

            def call(self, func, *args, **kwargs):
                if self.state == "OPEN":
                    if self._should_attempt_reset():
                        self.state = "HALF_OPEN"
                    else:
                        raise Exception("Circuit breaker is OPEN")

                try:
                    result = func(*args, **kwargs)
                    self._on_success()
                    return result
                except Exception as e:
                    self._on_failure()
                    raise e

            def _should_attempt_reset(self):
                if self.last_failure_time is None:
                    return False
                return (
                    datetime.now().timestamp() - self.last_failure_time
                ) > self.timeout

            def _on_success(self):
                self.failure_count = 0
                self.state = "CLOSED"

            def _on_failure(self):
                self.failure_count += 1
                self.last_failure_time = datetime.now().timestamp()
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"

        def failing_function():
            raise Exception("Service failure")

        def working_function():
            return "success"

        cb = CircuitBreaker(failure_threshold=2)

        # Test failures leading to open circuit
        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == "CLOSED"

        with pytest.raises(Exception):
            cb.call(failing_function)
        assert cb.state == "OPEN"

        # Test that circuit breaker blocks further calls
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            cb.call(working_function)


class TestDataStorageLogic:
    """Unit tests for data storage business logic"""

    @pytest.mark.unit
    def test_parquet_file_organization_logic(self):
        """Test parquet file organization business logic"""

        def generate_file_path(symbol, timeframe, date):
            """Generate parquet file path based on business rules"""
            year = date.year
            month = f"{date.month:02d}"
            day = f"{date.day:02d}"

            return f"data/{symbol}/{timeframe}/{year}/{month}/{day}.parquet"

        test_date = datetime(2024, 1, 15)
        file_path = generate_file_path("AAPL", "1min", test_date)

        expected_path = "data/AAPL/1min/2024/01/15.parquet"
        assert file_path == expected_path

        # Test different symbols and timeframes
        file_path_5min = generate_file_path("GOOGL", "5min", test_date)
        expected_path_5min = "data/GOOGL/5min/2024/01/15.parquet"
        assert file_path_5min == expected_path_5min

    @pytest.mark.unit
    def test_batch_processing_logic(self):
        """Test batch processing business logic"""

        def process_data_in_batches(data, batch_size):
            """Process data in batches to manage memory"""
            batches = []
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                batches.append(batch)
            return batches

        test_data = list(range(25))  # 25 items
        batches = process_data_in_batches(test_data, batch_size=10)

        assert len(batches) == 3  # 10, 10, 5
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

        # Test edge case: empty data
        empty_batches = process_data_in_batches([], batch_size=10)
        assert len(empty_batches) == 0

        # Test edge case: data smaller than batch size
        small_data = [1, 2, 3]
        small_batches = process_data_in_batches(small_data, batch_size=10)
        assert len(small_batches) == 1
        assert len(small_batches[0]) == 3

    @pytest.mark.unit
    def test_data_retention_logic(self):
        """Test data retention business logic"""

        def should_delete_file(file_date, retention_days):
            """Determine if a file should be deleted based on retention policy"""
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=retention_days)
            return file_date < cutoff_date

        # Test file within retention period
        recent_date = datetime.now(timezone.utc) - timedelta(days=5)
        assert not should_delete_file(recent_date, retention_days=30)

        # Test file outside retention period
        old_date = datetime.now(timezone.utc) - timedelta(days=45)
        assert should_delete_file(old_date, retention_days=30)

        # Test edge case: exactly at retention boundary
        boundary_date = datetime.now(timezone.utc) - timedelta(days=30, seconds=1)
        assert should_delete_file(boundary_date, retention_days=30)


class TestServiceHealthCheck:
    """Unit tests for service health check logic"""

    @pytest.mark.unit
    def test_component_health_check(self):
        """Test individual component health check logic"""

        def check_component_health(component_name, is_healthy, response_time=None):
            """Check individual component health"""
            status = "healthy" if is_healthy else "unhealthy"
            result = {
                "component": component_name,
                "status": status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if response_time is not None:
                result["response_time"] = response_time

            return result

        # Test healthy component
        healthy_result = check_component_health("database", True, 0.05)
        assert healthy_result["status"] == "healthy"
        assert healthy_result["component"] == "database"
        assert healthy_result["response_time"] == 0.05

        # Test unhealthy component
        unhealthy_result = check_component_health("redis", False)
        assert unhealthy_result["status"] == "unhealthy"
        assert unhealthy_result["component"] == "redis"

    @pytest.mark.unit
    def test_overall_service_health_logic(self):
        """Test overall service health determination logic"""

        def determine_overall_health(component_healths):
            """Determine overall service health from components"""
            if not component_healths:
                return "unknown"

            healthy_components = [
                c for c in component_healths if c.get("status") == "healthy"
            ]

            total_components = len(component_healths)
            healthy_ratio = len(healthy_components) / total_components

            if healthy_ratio == 1.0:
                return "healthy"
            elif healthy_ratio >= 0.5:
                return "degraded"
            else:
                return "unhealthy"

        # All healthy
        all_healthy = [
            {"component": "db", "status": "healthy"},
            {"component": "redis", "status": "healthy"},
            {"component": "api", "status": "healthy"},
        ]
        assert determine_overall_health(all_healthy) == "healthy"

        # Some unhealthy (degraded)
        some_unhealthy = [
            {"component": "db", "status": "healthy"},
            {"component": "redis", "status": "unhealthy"},
            {"component": "api", "status": "healthy"},
        ]
        assert determine_overall_health(some_unhealthy) == "degraded"

        # Majority unhealthy
        mostly_unhealthy = [
            {"component": "db", "status": "unhealthy"},
            {"component": "redis", "status": "unhealthy"},
            {"component": "api", "status": "healthy"},
        ]
        assert determine_overall_health(mostly_unhealthy) == "unhealthy"


class TestServiceMetrics:
    """Unit tests for service metrics collection logic"""

    @pytest.mark.unit
    def test_metrics_calculation_logic(self):
        """Test metrics calculation business logic"""

        def calculate_success_rate(total_requests, failed_requests):
            """Calculate success rate percentage"""
            if total_requests == 0:
                return 0.0
            return ((total_requests - failed_requests) / total_requests) * 100

        def calculate_average_response_time(response_times):
            """Calculate average response time"""
            if not response_times:
                return 0.0
            return sum(response_times) / len(response_times)

        def calculate_percentile(values, percentile):
            """Calculate percentile of values"""
            if not values:
                return 0.0
            sorted_values = sorted(values)
            index = int(len(sorted_values) * percentile / 100)
            return sorted_values[min(index, len(sorted_values) - 1)]

        # Test success rate calculation
        assert calculate_success_rate(100, 5) == 95.0
        assert calculate_success_rate(0, 0) == 0.0
        assert calculate_success_rate(10, 0) == 100.0

        # Test average response time
        response_times = [0.1, 0.2, 0.15, 0.3, 0.25]
        avg_time = calculate_average_response_time(response_times)
        assert abs(avg_time - 0.2) < 0.01  # Should be 0.2 seconds

        # Test percentile calculation
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p95 = calculate_percentile(values, 95)
        assert p95 == 10  # 95th percentile should be 10

    @pytest.mark.unit
    def test_rate_limiting_logic(self):
        """Test rate limiting business logic"""

        class RateLimiter:
            def __init__(self, max_requests, window_seconds):
                self.max_requests = max_requests
                self.window_seconds = window_seconds
                self.requests = []

            def is_allowed(self):
                now = datetime.now().timestamp()
                # Remove old requests outside the window
                self.requests = [
                    req_time
                    for req_time in self.requests
                    if now - req_time < self.window_seconds
                ]

                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return True
                return False

            def get_remaining_requests(self):
                now = datetime.now().timestamp()
                self.requests = [
                    req_time
                    for req_time in self.requests
                    if now - req_time < self.window_seconds
                ]
                return max(0, self.max_requests - len(self.requests))

        # Test rate limiter
        limiter = RateLimiter(max_requests=3, window_seconds=60)

        # Should allow first 3 requests
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True
        assert limiter.is_allowed() is True

        # Should block 4th request
        assert limiter.is_allowed() is False

        # Should have 0 remaining requests
        assert limiter.get_remaining_requests() == 0


class TestMarketHoursLogic:
    """Unit tests for market hours business logic"""

    @pytest.mark.unit
    def test_market_hours_validation(self):
        """Test market hours validation logic"""

        def is_market_open(timestamp, market_tz="America/New_York"):
            """Check if market is open at given timestamp"""
            # Convert to market timezone
            from zoneinfo import ZoneInfo

            market_time = timestamp.astimezone(ZoneInfo(market_tz))
            hour = market_time.hour
            minute = market_time.minute
            weekday = market_time.weekday()  # 0=Monday, 6=Sunday

            # Weekend check
            if weekday >= 5:  # Saturday or Sunday
                return False

            # Market hours: 9:30 AM - 4:00 PM ET
            market_open_minutes = 9 * 60 + 30  # 9:30 AM in minutes
            market_close_minutes = 16 * 60  # 4:00 PM in minutes
            current_minutes = hour * 60 + minute

            return market_open_minutes <= current_minutes < market_close_minutes

        # Test during market hours (Tuesday 10:00 AM ET)
        market_open_time = datetime(
            2024, 1, 9, 15, 0, tzinfo=timezone.utc
        )  # 10:00 AM ET
        assert is_market_open(market_open_time) is True

        # Test before market open (Tuesday 9:00 AM ET)
        before_open = datetime(2024, 1, 9, 14, 0, tzinfo=timezone.utc)  # 9:00 AM ET
        assert is_market_open(before_open) is False

        # Test after market close (Tuesday 5:00 PM ET)
        after_close = datetime(2024, 1, 9, 22, 0, tzinfo=timezone.utc)  # 5:00 PM ET
        assert is_market_open(after_close) is False

        # Test weekend (Saturday)
        weekend = datetime(
            2024, 1, 13, 15, 0, tzinfo=timezone.utc
        )  # Saturday 10:00 AM ET
        assert is_market_open(weekend) is False

    @pytest.mark.unit
    def test_trading_day_calculation(self):
        """Test trading day calculation logic"""

        def is_trading_day(date):
            """Check if a date is a trading day (weekday, not holiday)"""
            # Simple implementation - just check weekdays
            # In production, would also check for market holidays
            return date.weekday() < 5  # Monday=0, Friday=4

        def get_next_trading_day(date):
            """Get the next trading day after given date"""
            next_day = date + timedelta(days=1)
            while not is_trading_day(next_day):
                next_day += timedelta(days=1)
            return next_day

        # Test weekday
        tuesday = datetime(2024, 1, 9)  # Tuesday
        assert is_trading_day(tuesday) is True

        # Test weekend
        saturday = datetime(2024, 1, 13)  # Saturday
        assert is_trading_day(saturday) is False

        # Test next trading day from Friday
        friday = datetime(2024, 1, 12)  # Friday
        next_trading = get_next_trading_day(friday)
        assert next_trading.weekday() == 0  # Should be Monday
        assert next_trading.day == 15  # January 15, 2024 is Monday
