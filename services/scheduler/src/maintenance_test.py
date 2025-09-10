#!/usr/bin/env python3
"""
Comprehensive test script for the maintenance system.

This script validates all maintenance tasks, scheduling, monitoring,
and reporting capabilities of the trading scheduler's maintenance system.
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

# Import MaintenanceResult for type compatibility
from .maintenance import MaintenanceResult

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MaintenanceSystemTester:
    """Comprehensive tester for the maintenance system."""

    def __init__(self) -> None:
        self.test_results: dict = {}
        self.failed_tests: list = []
        self.passed_tests: list = []
        self.temp_dir: Optional[Path] = None
        self.mock_config: Optional[Any] = None
        self.redis_client: Optional[MockRedisClient] = None
        self.maintenance_manager = None
        self.maintenance_scheduler: Optional[Any] = None

    async def _cleanup_test_environment(self) -> bool:
        """Setup test environment with mock data and configurations."""
        try:
            logger.info("Setting up test environment...")

            # Create temporary directory for test data
            self.temp_dir = Path(tempfile.mkdtemp(prefix="maintenance_test_"))
            logger.info(f"Created test directory: {self.temp_dir}")

            # Create test data structure
            await self._create_test_data_structure()

            # Setup mock configuration
            await self._setup_mock_config()

            # Setup Redis client (using mock if needed)
            await self._setup_redis_client()

            # Initialize maintenance manager
            await self._initialize_maintenance_system()

            logger.info("Test environment setup completed")
            return True

        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            return False

    async def _create_test_data_structure(self) -> bool:
        """Create test data directories and files."""
        # Create directory structure
        directories = [
            "data/logs",
            "data/parquet/AAPL",
            "data/parquet/TSLA",
            "data/temp",
            "data/exports",
            "data/backups",
            "data/strategy_results",
        ]

        for dir_path in directories:
            if self.temp_dir is None:
                raise RuntimeError("Temp directory not initialized")
            full_path = self.temp_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)

        # Create test log files
        if self.temp_dir is None:
            raise RuntimeError("Temp directory not initialized")
        log_dir = self.temp_dir / "data/logs"

        # Large log file for rotation testing
        large_log = log_dir / "large_test.log"
        with open(large_log, "w") as f:
            for i in range(10000):
                f.write(
                    f"2024-01-{i % 30 + 1:02d} 12:00:00 - INFO - Test log entry {i}\n"
                )

        # Old log file for compression testing
        old_log = log_dir / "old_test.log"
        with open(old_log, "w") as f:
            f.write("2023-01-01 12:00:00 - INFO - Old log entry\n")

        # Set old timestamp
        old_timestamp = (datetime.now() - timedelta(days=10)).timestamp()
        old_log.touch()
        os.utime(old_log, (old_timestamp, old_timestamp))

        # Create test parquet files
        import pandas as pd

        for symbol in ["AAPL", "TSLA"]:
            if self.temp_dir is None:
                raise RuntimeError("Temp directory not initialized")
            symbol_dir = self.temp_dir / f"data/parquet/{symbol}"

            # Create multiple small files for consolidation testing
            for i in range(40):  # 40 days of data
                date = datetime.now() - timedelta(days=i)
                df = pd.DataFrame(
                    {
                        "timestamp": [date],
                        "symbol": [symbol],
                        "open": [100.0 + i],
                        "high": [105.0 + i],
                        "low": [95.0 + i],
                        "close": [102.0 + i],
                        "volume": [1000000],
                    }
                )

                filename = f"{date.strftime('%Y-%m-%d')}_1d.parquet"
                df.to_parquet(symbol_dir / filename, index=False)

        # Create temporary files
        if self.temp_dir is None:
            raise RuntimeError("Temp directory not initialized")
        temp_dir = self.temp_dir / "data/temp"
        for i in range(5):
            temp_file = temp_dir / f"temp_file_{i}.tmp"
            with open(temp_file, "w") as f:
                f.write(f"Temporary data {i}")

        # Create old export files
        if self.temp_dir is None:
            raise RuntimeError("Temp directory not initialized")
        export_dir = self.temp_dir / "data/exports"
        old_export = export_dir / "old_export.csv"
        with open(old_export, "w") as f:
            f.write("symbol,price\nAAPL,150.00\n")

        old_timestamp = (datetime.now() - timedelta(days=10)).timestamp()
        old_export.touch()
        os.utime(old_export, (old_timestamp, old_timestamp))

        logger.info("Test data structure created successfully")
        return True

    async def _setup_mock_config(self) -> None:
        """Setup mock configuration for testing."""

        class MockConfig:
            def __init__(self, temp_dir: Path):
                self.data = MockDataConfig(temp_dir)
                self.logging = MockLoggingConfig()
                self.backup = MockBackupConfig()
                self.redis = MockRedisConfig()

        class MockDataConfig:
            def __init__(self, temp_dir: Path):
                self.parquet_path = str(temp_dir / "data/parquet")
                self.retention_days = 30

        class MockLoggingConfig:
            def __init__(self) -> None:
                self.max_file_size = 1024 * 1024  # 1MB
                self.backup_count = 5

        class MockBackupConfig:
            def __init__(self) -> None:
                self.remote_enabled = False
                self.compression_level = 6

        class MockRedisConfig:
            def __init__(self) -> None:
                self.url = "redis://localhost:6379/0"
                self.host = "localhost"
                self.port = 6379
                self.database = 0
                self.password = None

        if self.temp_dir is None:
            raise RuntimeError("Temp directory not initialized")
        self.mock_config = MockConfig(self.temp_dir)
        logger.info("Mock configuration setup completed")

    async def _setup_redis_client(self) -> None:
        """Setup Redis client (mock or real)."""
        try:
            if self.mock_config is None:
                raise RuntimeError("Config not initialized")
            import redis.asyncio as redis

            self.redis_client = redis.from_url(self.mock_config.redis.url)

            # Test connection
            if self.redis_client:
                await self.redis_client.ping()
            logger.info("Redis connection established")

        except Exception as e:
            logger.warning(f"Redis connection failed, using mock: {e}")
            # Use mock Redis for testing
            self.redis_client = MockRedisClient()

    async def _initialize_maintenance_system(self) -> None:
        """Initialize the maintenance system."""
        try:
            from .maintenance import MaintenanceManager, MaintenanceScheduler

            if self.mock_config is None or self.redis_client is None:
                raise RuntimeError("Config or Redis client not initialized")
            self.maintenance_manager = MaintenanceManager(
                self.mock_config, self.redis_client
            )
            await self.maintenance_manager.register_tasks()

            self.maintenance_scheduler = MaintenanceScheduler(self.maintenance_manager)
            await self.maintenance_scheduler.start()

            logger.info("Maintenance system initialized")

        except Exception as e:
            logger.error(f"Failed to initialize maintenance system: {e}")
            raise

    async def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run all maintenance system tests."""
        logger.info("Starting comprehensive maintenance system tests...")

        test_suite = [
            ("setup", self._cleanup_test_environment),
            ("data_cleanup", self.test_data_cleanup_task),
            ("log_rotation", self.test_log_rotation_task),
            ("cache_cleanup", self.test_cache_cleanup_task),
            ("backup_task", self.test_backup_task),
            ("database_maintenance", self.test_database_maintenance_task),
            ("system_health_check", self.test_system_health_check_task),
            ("trading_data_maintenance", self.test_trading_data_maintenance_task),
            ("tradenote_export", self.test_tradenote_export_task),
            ("portfolio_reconciliation", self.test_portfolio_reconciliation_task),
            ("intelligent_maintenance", self.test_intelligent_maintenance_task),
            ("maintenance_scheduling", self.test_maintenance_scheduling),
            ("maintenance_monitoring", self.test_maintenance_monitoring),
            ("maintenance_reporting", self.test_maintenance_reporting),
            ("error_handling", self.test_error_handling),
            ("performance", self.test_performance),
        ]

        results = {}
        start_time = time.time()

        for test_name, test_func in test_suite:
            logger.info(f"Running test: {test_name}")
            test_start = time.time()  # Initialize before try block
            try:
                if self.maintenance_manager is None:
                    return {
                        test_name: {
                            "passed": False,
                            "duration": 0.0,
                            "error": "Maintenance manager not initialized",
                        }
                        for test_name, _ in test_suite
                    }
                result = await test_func()
                test_duration = time.time() - test_start

                results[test_name] = {
                    "passed": result,
                    "duration": test_duration,
                    "error": None,
                }

                if result:
                    self.passed_tests.append(test_name)
                    logger.info(f"✅ {test_name} passed ({test_duration:.2f}s)")
                else:
                    self.failed_tests.append(test_name)
                    logger.error(f"❌ {test_name} failed ({test_duration:.2f}s)")

            except Exception as e:
                test_duration = time.time() - test_start
                results[test_name] = {
                    "passed": False,
                    "duration": test_duration,
                    "error": str(e),
                }
                self.failed_tests.append(test_name)
                logger.error(f"❌ {test_name} failed with exception: {e}")

        total_duration = time.time() - start_time

        # Generate test summary
        await self._generate_test_summary(results, total_duration)

        # Cleanup
        await self.cleanup_test_environment()

        return results

    async def test_data_cleanup_task(self) -> bool:
        """Test data cleanup task functionality."""
        try:
            if self.maintenance_manager is None:
                return False
            result = await self.maintenance_manager.run_task("data_cleanup")

            # Verify files were processed
            success = (
                result.success
                and result.files_processed > 0
                and "cleanup" in result.message.lower()
            )

            # Check that old files were actually removed
            if self.temp_dir is None:
                return False
            old_export = self.temp_dir / "data/exports/old_export.csv"
            temp_files = list((self.temp_dir / "data/temp").glob("*.tmp"))

            cleanup_success = not old_export.exists() and len(temp_files) == 0

            return success and cleanup_success

        except Exception as e:
            logger.error(f"Data cleanup test failed: {e}")
            return False

    async def test_log_rotation_task(self) -> bool:
        """Test log rotation task functionality."""
        try:
            # Check initial state
            if self.temp_dir is None:
                return False
            log_dir = self.temp_dir / "data/logs"
            initial_files = list(log_dir.glob("*.log"))

            if self.maintenance_manager is None:
                return False
            result = await self.maintenance_manager.run_task("log_rotation")

            # Check that large log was rotated
            final_files = list(log_dir.glob("*.log*"))

            return (
                result.success
                and len(final_files) != len(initial_files)
                and "rotated" in result.message.lower()
            )

        except Exception as e:
            logger.error(f"Log rotation test failed: {e}")
            return False

    async def test_cache_cleanup_task(self) -> bool:
        """Test cache cleanup task functionality."""
        try:
            # Add some test cache data
            if self.redis_client is not None and hasattr(self.redis_client, "setex"):
                await self.redis_client.setex("test:cache:key1", 3600, "value1")
                await self.redis_client.setex("test:temp:key2", 3600, "value2")

            if self.maintenance_manager is None:
                return False
            result = await self.maintenance_manager.run_task("cache_cleanup")

            return result.success and "cleanup" in result.message.lower()

        except Exception as e:
            logger.error(f"Cache cleanup test failed: {e}")
            return False

    async def test_backup_task(self) -> bool:
        """Test backup task functionality."""
        try:
            if self.maintenance_manager is None:
                return False
            result = await self.maintenance_manager.run_task("backup_critical_data")

            # Check that backup directory was created
            if self.temp_dir is None:
                return False
            backup_dir = self.temp_dir / "data/backups"
            backup_files = (
                list(backup_dir.glob("*.tar.gz")) if backup_dir.exists() else []
            )

            return (
                result.success
                and len(backup_files) > 0
                and "backup" in result.message.lower()
            )

        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            return False
        return True

    async def test_database_maintenance_task(self) -> None:
        """Test database maintenance task functionality."""
        try:
            if self.maintenance_manager is None:
                return None
            result = await self.maintenance_manager.run_task("database_maintenance")

            return (
                result.success
                and "maintenance" in result.message.lower()
                and result.details is not None
            )

        except Exception as e:
            logger.error(f"Database maintenance test failed: {e}")
            return None

    async def test_system_health_check_task(self) -> bool:
        """Test system health check task functionality."""
        try:
            if self.maintenance_manager is None:
                return False
            result = await self.maintenance_manager.run_task("system_health_check")

            return (
                result.success
                and "health" in result.message.lower()
                and result.details is not None
                and "health_checks" in result.details
            )

        except Exception as e:
            logger.error(f"System health check test failed: {e}")
            return False

    async def test_trading_data_maintenance_task(self) -> bool:
        """Test trading data maintenance task functionality."""
        try:
            if self.maintenance_manager is None:
                return False
            if self.maintenance_manager is not None:
                result = await self.maintenance_manager.run_task(
                    "trading_data_maintenance"
                )
            else:
                return False

            # Check that parquet files were processed for consolidation
            if self.temp_dir is None:
                return False
            parquet_dir = self.temp_dir / "data/parquet"
            aapl_files = list((parquet_dir / "AAPL").glob("*.parquet"))

            return (
                result.success
                and "trading data" in result.message.lower()
                and len(aapl_files)
                > 0  # Files should still exist, possibly consolidated
            )

        except Exception as e:
            logger.error(f"Trading data maintenance test failed: {e}")
            return False

    async def test_tradenote_export_task(self) -> bool:
        """Test TradeNote export task functionality."""
        try:
            if self.maintenance_manager is None:
                return False
            result = await self.maintenance_manager.run_task("tradenote_export")

            # Check that export directory was created
            if self.temp_dir is None:
                return False
            export_dir = self.temp_dir / "data/exports/tradenote"

            return (
                result.success
                and "export" in result.message.lower()
                and export_dir.exists()
            )

        except Exception as e:
            logger.error(f"TradeNote export test failed: {e}")
            return False

    async def test_portfolio_reconciliation_task(self) -> bool:
        """Test portfolio reconciliation task functionality."""
        try:
            # Add some test position data
            if hasattr(self.redis_client, "setex"):
                position_data = {
                    "symbol": "AAPL",
                    "quantity": 100,
                    "avg_price": 150.00,
                    "timestamp": datetime.now().isoformat(),
                }
                if self.redis_client is not None:
                    await self.redis_client.setex(
                        "positions:AAPL", 3600, json.dumps(position_data)
                    )

            if self.maintenance_manager is not None:
                result = await self.maintenance_manager.run_task(
                    "portfolio_reconciliation"
                )
            else:
                return False

            return result.success and "reconciliation" in result.message.lower()

        except Exception as e:
            logger.error(f"Portfolio reconciliation test failed: {e}")
            return False

    async def test_intelligent_maintenance_task(self) -> bool:
        """Test intelligent maintenance task functionality."""
        try:
            if self.maintenance_manager is not None:
                result = await self.maintenance_manager.run_task(
                    "intelligent_maintenance"
                )
            else:
                return False

            return (
                result.success
                and "analysis" in result.message.lower()
                and result.details is not None
                and "performance_score" in result.details
            )

        except Exception as e:
            logger.error(f"Intelligent maintenance test failed: {e}")
            return False

    async def test_maintenance_scheduling(self) -> bool:
        """Test maintenance scheduling functionality."""
        try:
            if (
                not hasattr(self, "maintenance_scheduler")
                or self.maintenance_scheduler is None
            ):
                return False

            # Check that tasks are properly scheduled
            schedule = self.maintenance_scheduler.get_maintenance_schedule()

            expected_daily_tasks = [
                "daily_data_cleanup",
                "daily_log_rotation",
                "daily_cache_cleanup",
                "daily_system_health_check",
                "daily_tradenote_export",
                "daily_api_rate_limit_reset",
                "daily_intelligent_maintenance",
            ]

            scheduled_tasks = list(schedule.keys())
            daily_tasks_found = sum(
                1 for task in expected_daily_tasks if task in scheduled_tasks
            )

            return (
                daily_tasks_found >= len(expected_daily_tasks) * 0.8
            )  # At least 80% found

        except Exception as e:
            logger.error(f"Maintenance scheduling test failed: {e}")
            return False

    async def test_maintenance_monitoring(self) -> bool:
        """Test maintenance monitoring functionality."""
        try:
            if self.maintenance_manager is None or not hasattr(
                self.maintenance_manager, "monitor"
            ):
                return False

            monitor = self.maintenance_manager.monitor
            if monitor is None:
                return False

            # Test recording task execution
            test_result = MockMaintenanceResult()
            await monitor.record_task_start("test_task")
            await monitor.record_task_completion(test_result)

            # Test metrics retrieval
            metrics = await monitor.get_maintenance_metrics("test_task")

            return isinstance(metrics, dict)

        except Exception as e:
            logger.error(f"Maintenance monitoring test failed: {e}")
            return False

    async def test_maintenance_reporting(self) -> bool:
        """Test maintenance reporting functionality."""
        try:
            from .maintenance import MaintenanceReportGenerator

            if self.maintenance_manager is None:
                return False
            report_generator = MaintenanceReportGenerator(self.maintenance_manager)

            # Generate daily report
            daily_report = await report_generator.generate_daily_report()

            # Generate weekly report
            weekly_report = await report_generator.generate_weekly_report()

            return (
                isinstance(daily_report, dict)
                and isinstance(weekly_report, dict)
                and "summary" in daily_report
                and "trends" in weekly_report
            )

        except Exception as e:
            logger.error(f"Maintenance reporting test failed: {e}")
            return False

    async def test_error_handling(self) -> bool:
        """Test error handling in maintenance tasks."""
        try:
            # Test with invalid task name
            if self.maintenance_manager is None:
                return False
            result = await self.maintenance_manager.run_task("nonexistent_task")

            # Should fail gracefully
            if result.success:
                return False

            # Test task timeout handling
            # This would require a task that intentionally takes too long

            return not result.success and "not found" in result.message.lower()

        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False

    async def test_performance(self) -> bool:
        """Test maintenance system performance."""
        try:
            # Test multiple tasks running in sequence
            start_time = time.time()

            tasks_to_test = [
                "log_cleanup",
                "cache_cleanup",
                "data_cleanup",
                "system_health_check",
            ]

            for task_name in tasks_to_test:
                if self.maintenance_manager is None:
                    return False
                result = await self.maintenance_manager.run_task(task_name)
                if not result.success:
                    logger.warning(f"Performance test task {task_name} failed")

            total_duration = time.time() - start_time

            # Should complete within reasonable time (adjust as needed)
            return total_duration < 60.0  # 1 minute

        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False

    async def _generate_test_summary(
        self, results: Dict[str, Any], total_duration: float
    ) -> Dict[str, Any]:
        """Generate comprehensive test summary."""
        logger.info("\n" + "=" * 60)
        logger.info("MAINTENANCE SYSTEM TEST SUMMARY")
        logger.info("=" * 60)

        passed_count = len(self.passed_tests)
        failed_count = len(self.failed_tests)
        total_count = passed_count + failed_count

        logger.info(f"Total Tests: {total_count}")
        logger.info(f"Passed: {passed_count} (✅)")
        logger.info(f"Failed: {failed_count} (❌)")
        logger.info(f"Success Rate: {(passed_count / total_count * 100):.1f}%")
        logger.info(f"Total Duration: {total_duration:.2f}s")

        if self.failed_tests:
            logger.info("\nFailed Tests:")
            for test_name in self.failed_tests:
                error = results.get(test_name, {}).get("error")
                logger.info(f"  ❌ {test_name}" + (f" - {error}" if error else ""))

        if self.passed_tests:
            logger.info("\nPassed Tests:")
            for test_name in self.passed_tests:
                duration = results.get(test_name, {}).get("duration", 0)
                logger.info(f"  ✅ {test_name} ({duration:.2f}s)")

        # Export test results
        await self._export_test_results(results, total_duration)

        return {
            "total_tests": total_count,
            "passed": passed_count,
            "failed": failed_count,
            "success_rate": passed_count / total_count * 100 if total_count > 0 else 0,
            "duration": total_duration,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
        }

    async def _export_test_results(
        self, results: Dict[str, Any], total_duration: float
    ) -> str:
        """Export test results to file."""
        try:
            test_report = {
                "test_timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "summary": {
                    "total_tests": len(results),
                    "passed_tests": len(self.passed_tests),
                    "failed_tests": len(self.failed_tests),
                    "success_rate": (
                        (len(self.passed_tests) / len(results) * 100) if results else 0
                    ),
                },
                "detailed_results": results,
                "failed_tests": self.failed_tests,
                "passed_tests": self.passed_tests,
            }

            export_path = Path("data/test_results")
            export_path.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = export_path / f"maintenance_test_results_{timestamp}.json"

            with open(report_file, "w") as f:
                json.dump(test_report, f, indent=2, default=str)

            logger.info(f"Test results exported to: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"Failed to export test results: {e}")
            return ""

    async def cleanup_test_environment(self) -> None:
        """Clean up test environment."""
        try:
            if self.redis_client and hasattr(self.redis_client, "close"):
                await self.redis_client.close()

            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up test directory: {self.temp_dir}")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


class MockRedisClient:
    """Mock Redis client for testing when Redis is not available."""

    def __init__(self) -> None:
        self.data: Dict[str, str] = {}

    async def ping(self) -> bool:
        return True

    async def get(self, key: str) -> str | None:
        return self.data.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> bool:
        self.data[key] = value
        return True

    async def delete(self, key: str) -> int:
        if key in self.data:
            self.data.pop(key, None)
            return 1
        return 0

    async def keys(self, pattern: str = "*") -> list[str]:
        return [k for k in self.data.keys() if pattern.replace("*", "") in k]

    async def info(self) -> Dict[str, int]:
        return {
            "used_memory": 1024 * 1024,
            "connected_clients": 1,
            "keyspace_hits": 100,
            "keyspace_misses": 10,
        }

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        return []

    async def lpush(self, key: str, value: str) -> int:
        return 1

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        return True

    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        return len(mapping)

    async def zrange(
        self, key: str, start: int, end: int, withscores: bool = False
    ) -> list[str]:
        return []

    async def incr(self, key: str) -> int:
        return 1

    async def close(self) -> None:
        pass


class MockMaintenanceResult(MaintenanceResult):
    """Mock maintenance result for testing."""

    def __init__(self) -> None:
        super().__init__(
            task_name="test_task",
            success=True,
            duration=1.0,
            message="Test task completed",
            details={"test": True},
            files_processed=1,
            bytes_freed=1024,
        )


async def main() -> None:
    """Main test execution function."""
    logger.info("Starting maintenance system tests...")

    tester = MaintenanceSystemTester()
    results = await tester.run_all_tests()

    # Display test summary
    total_tests = len(results)
    passed_tests = sum(
        1 for r in results.values() if isinstance(r, dict) and r.get("passed", False)
    )
    logger.info(f"Test Summary: {passed_tests}/{total_tests} tests passed")

    # Exit with appropriate code
    failed_count = len(tester.failed_tests)
    if failed_count == 0:
        logger.info("🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error(f"💥 {failed_count} tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    # Run tests
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)
