#!/usr/bin/env python3
"""
Comprehensive test runner script for the AI Trading System.
Runs all tests with coverage analysis and generates reports.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class TestRunner:
    """Main test runner class."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_results: Dict[str, Dict] = {}
        self.coverage_results: Dict[str, Any] = {}
        self.start_time: Optional[float] = None

    def setup_environment(self):
        """Setup test environment."""
        print("🔧 Setting up test environment...")

        # Set test environment variables
        test_env = {
            "ENVIRONMENT": "test",
            "LOG_LEVEL": "DEBUG",
            "DB_NAME": "test_trading_system",
            "REDIS_DB": "1",
            "DISABLE_EXTERNAL_APIS": "true",
            "TEST_MODE": "true",
            "PYTHONPATH": str(self.project_root),
        }

        for key, value in test_env.items():
            os.environ[key] = value

        print(f"✅ Environment configured with {len(test_env)} variables")

    def run_unit_tests(self, verbose: bool = False) -> Dict:
        """Run unit tests with coverage."""
        print("\n🧪 Running unit tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/unit/",
            "--cov=services",
            "--cov=shared",
            "--cov=backtesting",
            "--cov-report=term-missing",
            "--cov-report=html:tests/coverage_html_unit",
            "--cov-report=xml:tests/coverage_unit.xml",
            "--cov-fail-under=80",
            "--tb=short",
            "--durations=10",
            "--maxfail=10",
            "-m",
            "unit",
            "--junitxml=tests/junit_unit.xml",
        ]

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        self.test_results["unit"] = {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }

        if result.returncode == 0:
            print(f"✅ Unit tests passed in {duration:.2f}s")
        else:
            print(f"❌ Unit tests failed (exit code: {result.returncode})")
            if verbose:
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")

        return self.test_results["unit"]

    def run_integration_tests(self, verbose: bool = False) -> Dict:
        """Run integration tests."""
        print("\n🔗 Running integration tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/integration/",
            "--tb=short",
            "--durations=10",
            "--maxfail=5",
            "-m",
            "integration",
            "--junitxml=tests/junit_integration.xml",
        ]

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        self.test_results["integration"] = {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }

        if result.returncode == 0:
            print(f"✅ Integration tests passed in {duration:.2f}s")
        else:
            print(f"❌ Integration tests failed (exit code: {result.returncode})")
            if verbose:
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")

        return self.test_results["integration"]

    def run_performance_tests(self, verbose: bool = False, quick: bool = False) -> Dict:
        """Run performance tests."""
        print("\n⚡ Running performance tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/performance/",
            "--tb=short",
            "--durations=20",
            "--maxfail=3",
            "--junitxml=tests/junit_performance.xml",
        ]

        if quick:
            cmd.extend(["-m", "performance and not slow"])
        else:
            cmd.extend(["-m", "performance"])

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        self.test_results["performance"] = {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }

        if result.returncode == 0:
            print(f"✅ Performance tests passed in {duration:.2f}s")
        else:
            print(f"❌ Performance tests failed (exit code: {result.returncode})")
            if verbose:
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")

        return self.test_results["performance"]

    def run_backtesting_tests(self, verbose: bool = False) -> Dict:
        """Run backtesting tests."""
        print("\n📊 Running backtesting tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/integration/test_backtesting.py",
            "--tb=short",
            "--durations=10",
            "--maxfail=5",
            "-m",
            "backtesting",
            "--junitxml=tests/junit_backtesting.xml",
        ]

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        self.test_results["backtesting"] = {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }

        if result.returncode == 0:
            print(f"✅ Backtesting tests passed in {duration:.2f}s")
        else:
            print(f"❌ Backtesting tests failed (exit code: {result.returncode})")
            if verbose:
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")

        return self.test_results["backtesting"]

    def run_smoke_tests(self, verbose: bool = False) -> Dict:
        """Run smoke tests for basic functionality."""
        print("\n💨 Running smoke tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/",
            "--tb=short",
            "--maxfail=1",
            "-m",
            "smoke",
            "--junitxml=tests/junit_smoke.xml",
        ]

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        self.test_results["smoke"] = {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }

        if result.returncode == 0:
            print(f"✅ Smoke tests passed in {duration:.2f}s")
        else:
            print(f"❌ Smoke tests failed (exit code: {result.returncode})")
            if verbose:
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")

        return self.test_results["smoke"]

    def run_security_tests(self, verbose: bool = False) -> Dict:
        """Run security tests."""
        print("\n🛡️ Running security tests...")

        cmd = [
            "python",
            "-m",
            "pytest",
            "tests/",
            "--tb=short",
            "--maxfail=3",
            "-m",
            "security",
            "--junitxml=tests/junit_security.xml",
        ]

        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        self.test_results["security"] = {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }

        if result.returncode == 0:
            print(f"✅ Security tests passed in {duration:.2f}s")
        else:
            print(f"❌ Security tests failed (exit code: {result.returncode})")
            if verbose:
                print(f"STDOUT:\n{result.stdout}")
                print(f"STDERR:\n{result.stderr}")

        return self.test_results["security"]

    def analyze_coverage(self) -> Dict:
        """Analyze test coverage from generated reports."""
        print("\n📊 Analyzing test coverage...")

        coverage_file = self.project_root / "tests" / "coverage_unit.xml"

        if not coverage_file.exists():
            print("⚠️ Coverage report not found")
            return {}

        try:
            import xml.etree.ElementTree as ET

            tree = ET.parse(coverage_file)
            root: ET.Element = tree.getroot()

            # Extract coverage metrics
            coverage_data: Dict[str, Dict[str, Any]] = {}

            for package in root.findall(".//package"):
                package_name = package.get("name", "unknown")
                line_rate = float(package.get("line-rate", 0)) * 100
                branch_rate = float(package.get("branch-rate", 0)) * 100

                coverage_data[package_name] = {
                    "line_coverage": line_rate,
                    "branch_coverage": branch_rate,
                }

            # Overall coverage
            overall_line_rate = float(root.get("line-rate", 0)) * 100
            overall_branch_rate = float(root.get("branch-rate", 0)) * 100

            self.coverage_results = {
                "overall_line_coverage": overall_line_rate,
                "overall_branch_coverage": overall_branch_rate,
                "package_coverage": coverage_data,
            }

            print(f"📈 Overall line coverage: {overall_line_rate:.1f}%")
            print(f"📈 Overall branch coverage: {overall_branch_rate:.1f}%")

            # Check coverage threshold
            if overall_line_rate >= 80:
                print("✅ Coverage threshold (80%) achieved!")
            else:
                print(f"❌ Coverage below threshold: {overall_line_rate:.1f}% < 80%")

            return self.coverage_results

        except Exception as e:
            print(f"⚠️ Failed to analyze coverage: {e}")
            return {}

    def run_linting(self, verbose: bool = False) -> Dict:
        """Run code linting and quality checks."""
        print("\n🔍 Running code quality checks...")

        # Run flake8
        flake8_cmd = [
            "flake8",
            "services/",
            "shared/",
            "backtesting/",
            "--max-line-length=120",
        ]
        flake8_result = subprocess.run(
            flake8_cmd, cwd=self.project_root, capture_output=True, text=True
        )

        # Run black check
        black_cmd = [
            "black",
            "--check",
            "--diff",
            "services/",
            "shared/",
            "backtesting/",
        ]
        black_result = subprocess.run(
            black_cmd, cwd=self.project_root, capture_output=True, text=True
        )

        # Run isort check
        isort_cmd = [
            "isort",
            "--check-only",
            "--diff",
            "services/",
            "shared/",
            "backtesting/",
        ]
        isort_result = subprocess.run(
            isort_cmd, cwd=self.project_root, capture_output=True, text=True
        )

        lint_results = {
            "flake8": {
                "exit_code": flake8_result.returncode,
                "output": flake8_result.stdout + flake8_result.stderr,
                "success": flake8_result.returncode == 0,
            },
            "black": {
                "exit_code": black_result.returncode,
                "output": black_result.stdout + black_result.stderr,
                "success": black_result.returncode == 0,
            },
            "isort": {
                "exit_code": isort_result.returncode,
                "output": isort_result.stdout + isort_result.stderr,
                "success": isort_result.returncode == 0,
            },
        }

        all_passed = all(result["success"] for result in lint_results.values())

        if all_passed:
            print("✅ All linting checks passed")
        else:
            if not all(result["success"] for result in lint_results.values()):
                print("❌ Some linting checks failed:")
                for tool, result in lint_results.items():
                    if not result["success"]:
                        output = result.get("output", "")
                        if isinstance(output, str):
                            print(f"  {tool}: {output[:200]}...")
                        else:
                            print(f"  {tool}: {str(output)[:200]}...")

        return lint_results

    def run_type_checking(self, verbose: bool = False) -> Dict:
        """Run type checking with mypy."""
        print("\n🏷️ Running type checking...")

        cmd = [
            "mypy",
            "services/",
            "shared/",
            "backtesting/",
            "--ignore-missing-imports",
            "--disallow-untyped-defs",
            "--warn-return-any",
            "--warn-unused-configs",
            "--show-error-codes",
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        type_check_result = {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "output": result.stdout + result.stderr,
            "success": result.returncode == 0,
        }

        if result.returncode == 0:
            print(f"✅ Type checking passed in {duration:.2f}s")
        else:
            print("❌ Type checking failed:")
            if verbose:
                print(f"Output:\n{result.stdout}")

        return type_check_result

    def run_security_scan(self, verbose: bool = False) -> Dict:
        """Run security scanning with bandit."""
        print("\n🔒 Running security scan...")

        cmd = [
            "bandit",
            "-r",
            "services/",
            "shared/",
            "backtesting/",
            "-f",
            "json",
            "-o",
            "tests/security_report.json",
        ]

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        security_result = {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "output": result.stdout + result.stderr,
            "success": result.returncode == 0,
        }

        # Parse security report
        security_report_path = self.project_root / "tests" / "security_report.json"
        if security_report_path.exists():
            try:
                with open(security_report_path, "r") as f:
                    security_data = json.load(f)

                high_severity = len(
                    [
                        issue
                        for issue in security_data.get("results", [])
                        if issue.get("issue_severity") == "HIGH"
                    ]
                )
                medium_severity = len(
                    [
                        issue
                        for issue in security_data.get("results", [])
                        if issue.get("issue_severity") == "MEDIUM"
                    ]
                )

                security_result["high_severity_issues"] = high_severity
                security_result["medium_severity_issues"] = medium_severity

                if high_severity == 0:
                    print(
                        f"✅ Security scan passed: {medium_severity} medium issues found"
                    )
                else:
                    print(
                        f"❌ Security scan found {high_severity} high severity issues"
                    )

            except Exception as e:
                print(f"⚠️ Failed to parse security report: {e}")

        return security_result

    def run_dependency_check(self) -> Dict:
        """Check for known vulnerabilities in dependencies."""
        print("\n📦 Checking dependencies for vulnerabilities...")

        cmd = ["safety", "check", "--json"]

        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        dependency_result = {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "output": result.stdout + result.stderr,
            "success": result.returncode == 0,
        }

        # Parse safety results
        try:
            if result.stdout:
                safety_data = json.loads(result.stdout)
                vulnerability_count = len(safety_data)
                dependency_result["vulnerabilities_found"] = vulnerability_count

                if vulnerability_count == 0:
                    print("✅ No known vulnerabilities in dependencies")
                else:
                    print(
                        f"❌ Found {vulnerability_count} vulnerabilities in dependencies"
                    )
            else:
                print("✅ Dependency check passed")

        except json.JSONDecodeError:
            print("⚠️ Could not parse safety output")

        return dependency_result

    def generate_test_report(self) -> None:
        """Generate comprehensive test report."""
        print("\n📋 Generating test report...")

        report: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_duration_seconds": (
                time.time() - self.start_time if self.start_time else 0
            ),
            "test_results": self.test_results,
            "coverage_results": self.coverage_results,
            "summary": {
                "tests_passed": sum(
                    1 for result in self.test_results.values() if result.get("success")
                ),
                "tests_failed": sum(
                    1
                    for result in self.test_results.values()
                    if not result.get("success")
                ),
                "total_test_categories": len(self.test_results),
                "overall_success": all(
                    result.get("success", False)
                    for result in self.test_results.values()
                ),
            },
        }

        # Save report
        report_path = self.project_root / "tests" / "test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📄 Test report saved to: {report_path}")

        # Print summary
        summary: Dict[str, Any] = report["summary"]
        print("\n📊 Test Summary:")
        print(
            f"  Categories passed: {summary['tests_passed']}/{summary['total_test_categories']}"
        )
        print(
            f"  Overall success: {'✅ YES' if summary['overall_success'] else '❌ NO'}"
        )
        print(f"  Total duration: {report['total_duration_seconds']:.2f}s")

        if self.coverage_results:
            line_coverage = self.coverage_results.get("overall_line_coverage", 0)
            print(f"  Line coverage: {line_coverage:.1f}%")

    def create_docker_test_runner(self) -> str:
        """Create Docker-based test runner script."""
        docker_script = """#!/bin/bash
set -e

echo "🐳 Running tests in Docker containers..."

# Start test infrastructure
docker-compose -f docker-compose.test.yml up -d postgres redis

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Run tests in container
docker run --rm \\
    --network full-ai-trader_trading_network \\
    -v $(pwd):/app \\
    -w /app \\
    -e ENVIRONMENT=test \\
    -e DB_HOST=postgres \\
    -e REDIS_HOST=redis \\
    -e DB_NAME=test_trading_system \\
    -e REDIS_DB=1 \\
    python:3.11-slim \\
    bash -c "
        pip install -r requirements.txt &&
        pip install pytest pytest-cov pytest-asyncio &&
        python scripts/run_tests.py --all --verbose
    "

# Cleanup
docker-compose -f docker-compose.test.yml down

echo "🏁 Docker tests completed"
"""

        docker_script_path = self.project_root / "scripts" / "run_tests_docker.sh"
        with open(docker_script_path, "w") as f:
            f.write(docker_script)

        # Make executable
        os.chmod(docker_script_path, 0o755)

        print(f"🐳 Docker test script created: {docker_script_path}")
        return str(docker_script_path)

    def run_all_tests(self, verbose: bool = False, quick: bool = False) -> bool:
        """Run all test categories."""
        print("🚀 Starting comprehensive test suite...")
        self.start_time = time.time()

        # Setup environment
        self.setup_environment()

        test_categories = [
            ("smoke", lambda: self.run_smoke_tests(verbose)),
            ("unit", lambda: self.run_unit_tests(verbose)),
            ("integration", lambda: self.run_integration_tests(verbose)),
            ("performance", lambda: self.run_performance_tests(verbose, quick)),
            ("backtesting", lambda: self.run_backtesting_tests(verbose)),
            ("security", lambda: self.run_security_tests(verbose)),
        ]

        # Run each test category
        for category_name, test_func in test_categories:
            try:
                result = test_func()
                if not result.get("success", False):
                    print(f"⚠️ {category_name.title()} tests failed but continuing...")
            except Exception as e:
                print(f"❌ Error running {category_name} tests: {e}")
                self.test_results[category_name] = {"success": False, "error": str(e)}

        # Analyze coverage
        self.analyze_coverage()

        # Generate report
        self.generate_test_report()

        # Return overall success
        overall_success = all(
            result.get("success", False) for result in self.test_results.values()
        )

        if overall_success:
            print("\n🎉 All tests passed!")
        else:
            print("\n💥 Some tests failed. Check the report for details.")

        return overall_success

    def run_parallel_tests(self, verbose: bool = False) -> bool:
        """Run tests in parallel for faster execution."""
        print("⚡ Running tests in parallel...")

        import concurrent.futures

        # Define test categories that can run in parallel
        parallel_tests = [
            (
                "unit_parallel",
                lambda: self.run_command(
                    [
                        "python",
                        "-m",
                        "pytest",
                        "tests/unit/",
                        "--cov=services",
                        "--cov=shared",
                        "-n",
                        "auto",  # pytest-xdist for parallel execution
                        "--tb=short",
                        "-m",
                        "unit and not database",
                    ]
                ),
            ),
            (
                "integration_parallel",
                lambda: self.run_command(
                    [
                        "python",
                        "-m",
                        "pytest",
                        "tests/integration/",
                        "-n",
                        "2",  # Limited parallelism for integration tests
                        "--tb=short",
                        "-m",
                        "integration and not slow",
                    ]
                ),
            ),
        ]

        # Run tests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_test = {
                executor.submit(test_func): test_name
                for test_name, test_func in parallel_tests
            }

            for future in concurrent.futures.as_completed(future_to_test):
                test_name = future_to_test[future]
                try:
                    result = future.result()
                    self.test_results[test_name] = result
                    status = "✅ PASSED" if result["success"] else "❌ FAILED"
                    print(f"{status} {test_name} in {result['duration_seconds']:.2f}s")
                except Exception as e:
                    print(f"❌ {test_name} raised exception: {e}")
                    self.test_results[test_name] = {"success": False, "error": str(e)}

        return all(
            result.get("success", False) for result in self.test_results.values()
        )

    def run_command(self, cmd: List[str]) -> Dict:
        """Run a command and return results."""
        start_time = time.time()
        result = subprocess.run(
            cmd, cwd=self.project_root, capture_output=True, text=True
        )
        duration = time.time() - start_time

        return {
            "exit_code": result.returncode,
            "duration_seconds": duration,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0,
        }

    def create_test_environment_check(self) -> bool:
        """Check if test environment is properly configured."""
        print("🔍 Checking test environment...")

        checks = [
            ("Python version", lambda: sys.version_info >= (3, 9)),
            ("Project root exists", lambda: self.project_root.exists()),
            ("Tests directory exists", lambda: (self.project_root / "tests").exists()),
            (
                "Services directory exists",
                lambda: (self.project_root / "services").exists(),
            ),
            ("Shared module exists", lambda: (self.project_root / "shared").exists()),
            ("pytest.ini exists", lambda: (self.project_root / "pytest.ini").exists()),
        ]

        all_passed = True
        for check_name, check_func in checks:
            try:
                passed = check_func()
                status = "✅" if passed else "❌"
                print(f"  {status} {check_name}")
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"  ❌ {check_name}: {e}")
                all_passed = False

        if all_passed:
            print("✅ Test environment ready")
        else:
            print("❌ Test environment has issues")

        return all_passed

    def cleanup_test_artifacts(self):
        """Clean up test artifacts and temporary files."""
        print("\n🧹 Cleaning up test artifacts...")

        cleanup_patterns = [
            "tests/**/__pycache__",
            "tests/**/*.pyc",
            "tests/**/.pytest_cache",
            "tests/coverage_html*",
            "tests/*.xml",
            "tests/*.json",
            ".coverage*",
            "**/.pytest_cache",
        ]

        import glob

        cleaned_count = 0

        for pattern in cleanup_patterns:
            for path in glob.glob(str(self.project_root / pattern), recursive=True):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        cleaned_count += 1
                    elif os.path.isdir(path):
                        import shutil

                        shutil.rmtree(path)
                        cleaned_count += 1
                except Exception as e:
                    print(f"⚠️ Could not clean {path}: {e}")

        print(f"🗑️ Cleaned {cleaned_count} test artifacts")

    def handle_utility_operations(self, args) -> Optional[int]:
        """Handle utility operations like cleanup, environment check, etc."""
        if args.check_env:
            return self._handle_env_check()

        if args.cleanup:
            return self._handle_cleanup()

        if args.docker:
            return self._handle_docker_script()

        if args.coverage_only:
            return self._handle_coverage_only()

        return None

    def _handle_env_check(self) -> int:
        """Handle environment check operation."""
        if self.create_test_environment_check():
            print("✅ Environment check passed")
            return 0
        else:
            print("❌ Environment check failed")
            return 1

    def _handle_cleanup(self) -> int:
        """Handle cleanup operation."""
        self.cleanup_test_artifacts()
        return 0

    def _handle_docker_script(self) -> int:
        """Handle Docker script creation."""
        docker_script_path = self.create_docker_test_runner()
        print(f"✅ Docker test script created: {docker_script_path}")
        return 0

    def _handle_coverage_only(self) -> int:
        """Handle coverage-only analysis."""
        self.analyze_coverage()
        return 0

    def handle_code_quality_checks(self, args) -> Optional[int]:
        """Handle code quality checks like linting, type checking, security scans."""
        if args.lint:
            return self._handle_lint_check(args.verbose)

        if args.type_check:
            return self._handle_type_check(args.verbose)

        if args.security_scan:
            return self._handle_security_scan(args.verbose)

        return None

    def _handle_lint_check(self, verbose: bool) -> int:
        """Handle linting check."""
        lint_results = self.run_linting(verbose)
        return 0 if all(r["success"] for r in lint_results.values()) else 1

    def _handle_type_check(self, verbose: bool) -> int:
        """Handle type checking."""
        type_result = self.run_type_checking(verbose)
        return 0 if type_result["success"] else 1

    def _handle_security_scan(self, verbose: bool) -> int:
        """Handle security scan."""
        security_result = self.run_security_scan(verbose)
        dependency_result = self.run_dependency_check()
        return 0 if security_result["success"] and dependency_result["success"] else 1

        return None

    def run_tests_based_on_args(self, args) -> bool:
        """Run tests based on command line arguments."""
        if args.parallel:
            return self.run_parallel_tests(args.verbose)

        if args.all:
            return self.run_all_tests(args.verbose, args.quick)

        return self._run_individual_tests(args)

    def _run_individual_tests(self, args) -> bool:
        """Run individual test categories based on arguments."""
        test_methods = {
            "smoke": lambda: self.run_smoke_tests(args.verbose),
            "unit": lambda: self.run_unit_tests(args.verbose),
            "integration": lambda: self.run_integration_tests(args.verbose),
            "performance": lambda: self.run_performance_tests(args.verbose, args.quick),
            "backtesting": lambda: self.run_backtesting_tests(args.verbose),
            "security": lambda: self.run_security_tests(args.verbose),
        }

        success = True
        tests_run = False

        for test_type, method in test_methods.items():
            if getattr(args, test_type):
                result = method()
                success = success and result["success"]
                tests_run = True

        # If no specific category selected, run smoke tests by default
        if not tests_run:
            print("ℹ️ No specific test category selected, running smoke tests...")
            result = self.run_smoke_tests(args.verbose)
            success = result["success"]

        return success


def create_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for AI Trading System"
    )

    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests only"
    )
    parser.add_argument(
        "--backtesting", action="store_true", help="Run backtesting tests only"
    )
    parser.add_argument(
        "--security", action="store_true", help="Run security tests only"
    )
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests only")
    parser.add_argument("--all", action="store_true", help="Run all test categories")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick tests (skip slow tests)"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage-only",
        action="store_true",
        help="Only analyze coverage from existing reports",
    )
    parser.add_argument(
        "--lint", action="store_true", help="Run linting and code quality checks"
    )
    parser.add_argument("--type-check", action="store_true", help="Run type checking")
    parser.add_argument(
        "--security-scan", action="store_true", help="Run security scan"
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Clean up test artifacts"
    )
    parser.add_argument(
        "--check-env", action="store_true", help="Check test environment"
    )
    parser.add_argument(
        "--docker", action="store_true", help="Create Docker test runner script"
    )
    parser.add_argument("--project-root", default=".", help="Project root directory")

    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Determine project root
    project_root = os.path.abspath(args.project_root)
    runner = TestRunner(project_root)

    # Handle utility operations first
    utility_result = runner.handle_utility_operations(args)
    if utility_result is not None:
        return utility_result

    # Handle code quality checks
    quality_result = runner.handle_code_quality_checks(args)
    if quality_result is not None:
        return quality_result

    # Run tests based on arguments
    success = runner.run_tests_based_on_args(args)

    # Generate final report if any tests were run
    if runner.test_results:
        runner.analyze_coverage()
        runner.generate_test_report()

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
