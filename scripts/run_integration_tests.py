#!/usr/bin/env python3
"""
Integration Test Management Script

This script manages the complete lifecycle of integration tests including:
- Environment validation and setup
- Docker infrastructure startup/shutdown
- Test execution with real services
- Result reporting and cleanup
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Set up project path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            project_root
            / "integration_test_data"
            / "logs"
            / "integration_test_runner.log"
        ),
    ],
)
logger = logging.getLogger(__name__)


class IntegrationTestRunner:
    """Manages integration test lifecycle."""

    def __init__(self, args):
        self.args = args
        self.project_root = project_root
        self.test_data_dir = self.project_root / "integration_test_data"
        self.compose_file = self.project_root / "docker-compose.integration.yml"
        self.env_file = self.project_root / ".env.integration"

        # Ensure test data directory exists
        self.test_data_dir.mkdir(exist_ok=True)
        (self.test_data_dir / "logs").mkdir(exist_ok=True)
        (self.test_data_dir / "data").mkdir(exist_ok=True)
        (self.test_data_dir / "backups").mkdir(exist_ok=True)
        (self.test_data_dir / "postgres").mkdir(exist_ok=True)
        (self.test_data_dir / "redis").mkdir(exist_ok=True)

    def run(self) -> bool:
        """Run the complete integration test suite."""
        logger.info("🚀 Starting Integration Test Runner")
        logger.info("=" * 60)

        try:
            # Step 1: Validate environment
            if not self._validate_environment():
                logger.error("❌ Environment validation failed")
                return False

            # Step 2: Setup infrastructure
            if not self._setup_infrastructure():
                logger.error("❌ Infrastructure setup failed")
                return False

            # Step 3: Wait for infrastructure to be ready
            if not self._wait_for_infrastructure():
                logger.error("❌ Infrastructure readiness check failed")
                return False

            # Step 4: Run tests
            test_result = self._run_tests()

            # Step 5: Generate report
            self._generate_report(test_result)

            return test_result

        except KeyboardInterrupt:
            logger.warning("⚠️ Integration test interrupted by user")
            return False
        except Exception as e:
            logger.error(f"💥 Integration test failed with exception: {e}")
            return False
        finally:
            # Always cleanup if requested
            if not self.args.no_cleanup:
                self._cleanup_infrastructure()

    def _validate_environment(self) -> bool:
        """Validate that the environment is ready for integration tests."""
        logger.info("🔍 Validating integration test environment...")

        # Check required files exist
        required_files = [
            self.env_file,
            self.compose_file,
            self.project_root / "data" / "parquet",
        ]

        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Required file/directory not found: {file_path}")
                return False

        # Check Docker is available
        try:
            result = subprocess.run(
                ["docker", "--version"], capture_output=True, text=True, check=True
            )
            logger.info(f"Docker version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Docker not found or not working")
            return False

        # Check Docker Compose is available
        try:
            result = subprocess.run(
                ["docker", "compose", "version"],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"Docker Compose version: {result.stdout.strip()}")
        except subprocess.CalledProcessError:
            logger.error("Docker Compose not found or not working")
            return False

        # Validate credentials are different from production
        self._validate_credentials()

        # Check data availability
        data_path = self.project_root / "data" / "parquet" / "market_data"
        if data_path.exists():
            symbol_count = len(list(data_path.iterdir()))
            logger.info(f"Found historical data for {symbol_count} symbols")
        else:
            logger.warning("No historical market data found")

        logger.info("✅ Environment validation passed")
        return True

    def _validate_credentials(self):
        """Validate that integration credentials are different from production."""
        # Load integration environment
        env_vars = {}
        if self.env_file.exists():
            with open(self.env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key] = value

        # Check database name
        db_name = env_vars.get("DB_NAME", "")
        if not any(
            test_pattern in db_name.lower() for test_pattern in ["test", "integration"]
        ):
            raise ValueError(f"Database name '{db_name}' doesn't indicate testing")

        logger.info("✅ Credential validation passed")

    def _setup_infrastructure(self) -> bool:
        """Set up Docker infrastructure for integration tests."""
        logger.info("🏗️ Setting up integration test infrastructure...")

        try:
            # Stop any existing containers
            self._run_docker_compose(["down", "--remove-orphans"], check=False)

            # Pull latest images
            if not self.args.skip_build:
                logger.info("📦 Pulling/building Docker images...")
                self._run_docker_compose(["build", "--no-cache"])

            # Start infrastructure services
            logger.info("🚀 Starting infrastructure services...")
            self._run_docker_compose(
                ["up", "-d", "postgres_integration", "redis_integration"]
            )

            logger.info("✅ Infrastructure setup completed")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Infrastructure setup failed: {e}")
            return False

    def _wait_for_infrastructure(self) -> bool:
        """Wait for infrastructure services to be ready."""
        logger.info("⏳ Waiting for infrastructure to be ready...")

        services = ["postgres_integration", "redis_integration"]
        timeout = 120  # 2 minutes
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check service health
                result = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-f",
                        str(self.compose_file),
                        "ps",
                        "--format",
                        "json",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                containers = []
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        containers.append(json.loads(line))

                healthy_services = []
                for container in containers:
                    if container["Service"] in services:
                        if container["State"] == "running":
                            # Additional health check using docker inspect
                            inspect_result = subprocess.run(
                                [
                                    "docker",
                                    "inspect",
                                    container["Name"],
                                    "--format",
                                    "{{.State.Health.Status}}",
                                ],
                                capture_output=True,
                                text=True,
                            )

                            health_status = inspect_result.stdout.strip()
                            if health_status in [
                                "healthy",
                                "",
                            ]:  # Empty means no healthcheck
                                healthy_services.append(container["Service"])

                logger.info(f"Healthy services: {healthy_services}/{len(services)}")

                if len(healthy_services) == len(services):
                    logger.info("✅ All infrastructure services are ready")
                    return True

                time.sleep(5)

            except subprocess.CalledProcessError as e:
                logger.warning(f"Error checking service health: {e}")
                time.sleep(5)

        logger.error(f"⏰ Infrastructure not ready after {timeout} seconds")
        return False

    def _run_tests(self) -> bool:
        """Run the integration tests."""
        logger.info("🧪 Running integration tests...")

        test_results = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "errors": [],
            "duration": 0,
        }

        start_time = time.time()

        try:
            # Build test arguments
            test_args = [
                "docker",
                "compose",
                "-f",
                str(self.compose_file),
                "run",
                "--rm",
                "-T",
                "-e",
                f'PYTEST_ARGS={self.args.pytest_args or ""}',
                "integration_test_runner",
            ]

            if self.args.verbose:
                test_args.extend(["-v"])

            # Run tests
            logger.info(f"Executing: {' '.join(test_args)}")

            result = subprocess.run(
                test_args,
                cwd=self.project_root,
                env={
                    **os.environ,
                    "COMPOSE_PROJECT_NAME": "trading_system_integration",
                },
                timeout=self.args.timeout,
            )

            test_results["duration"] = time.time() - start_time
            test_results["exit_code"] = result.returncode

            if result.returncode == 0:
                logger.info("✅ All integration tests passed!")
                test_results["tests_passed"] = 1  # Simplified for now
                return True
            else:
                logger.error(
                    f"❌ Integration tests failed with exit code: {result.returncode}"
                )
                test_results["tests_failed"] = 1
                test_results["errors"].append(
                    f"Test runner exit code: {result.returncode}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Tests timed out after {self.args.timeout} seconds")
            test_results["errors"].append(
                f"Tests timed out after {self.args.timeout} seconds"
            )
            return False

        except Exception as e:
            logger.error(f"💥 Test execution failed: {e}")
            test_results["errors"].append(str(e))
            return False

        finally:
            test_results["end_time"] = datetime.now(timezone.utc).isoformat()

            # Save test results
            results_file = self.test_data_dir / "logs" / "test_results.json"
            with open(results_file, "w") as f:
                json.dump(test_results, f, indent=2)

    def _generate_report(self, test_passed: bool):
        """Generate integration test report."""
        logger.info("📊 Generating integration test report...")

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_passed": test_passed,
            "environment": "integration",
            "infrastructure": {
                "postgres": self._check_service_status("postgres_integration"),
                "redis": self._check_service_status("redis_integration"),
            },
            "test_configuration": {
                "timeout": self.args.timeout,
                "cleanup": not self.args.no_cleanup,
                "verbose": self.args.verbose,
                "skip_build": self.args.skip_build,
            },
        }

        # Save report
        report_file = (
            self.test_data_dir
            / "logs"
            / f"integration_test_report_{int(time.time())}.json"
        )
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("INTEGRATION TEST REPORT")
        print("=" * 60)
        print(f"Status: {'✅ PASSED' if test_passed else '❌ FAILED'}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Report saved: {report_file}")
        print("=" * 60)

        logger.info(f"Report generated: {report_file}")

    def _check_service_status(self, service_name: str) -> Dict[str, Any]:
        """Check the status of a Docker service."""
        try:
            result = subprocess.run(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(self.compose_file),
                    "ps",
                    service_name,
                    "--format",
                    "json",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stdout.strip():
                container_info = json.loads(result.stdout.strip())
                return {
                    "status": container_info.get("State", "unknown"),
                    "name": container_info.get("Name", "unknown"),
                }
            else:
                return {"status": "not_found"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _cleanup_infrastructure(self):
        """Clean up Docker infrastructure."""
        logger.info("🧹 Cleaning up integration test infrastructure...")

        try:
            # Stop and remove containers
            self._run_docker_compose(["down"], check=False)

            # Remove test data if requested
            if self.args.clean_data:
                import shutil

                if self.test_data_dir.exists():
                    shutil.rmtree(self.test_data_dir)
                    logger.info("🗑️ Test data directory cleaned")

            logger.info("✅ Cleanup completed")

        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def _run_docker_compose(self, args: List[str], check: bool = True):
        """Run docker compose command."""
        cmd = [
            "docker",
            "compose",
            "-f",
            str(self.compose_file),
            "--env-file",
            str(self.env_file),
        ] + args

        env = os.environ.copy()
        env["COMPOSE_PROJECT_NAME"] = "trading_system_integration"

        # Load integration environment variables
        if self.env_file.exists():
            with open(self.env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env[key.strip()] = value.strip()

        logger.debug(f"Running: {' '.join(cmd)}")

        return subprocess.run(cmd, cwd=self.project_root, env=env, check=check)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Integration Test Runner")

    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Test timeout in seconds (default: 1800)",
    )

    parser.add_argument(
        "--no-cleanup", action="store_true", help="Skip cleanup after tests"
    )

    parser.add_argument(
        "--clean-data",
        action="store_true",
        help="Clean test data directory during cleanup",
    )

    parser.add_argument(
        "--skip-build", action="store_true", help="Skip Docker image building"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument(
        "--pytest-args", type=str, help="Additional arguments to pass to pytest"
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick integration tests only"
    )

    args = parser.parse_args()

    # Adjust settings for quick mode
    if args.quick:
        args.timeout = 600  # 10 minutes
        args.pytest_args = (args.pytest_args or "") + ' -m "not slow"'

    # Create and run the integration test runner
    runner = IntegrationTestRunner(args)
    success = runner.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
