#!/usr/bin/env python3
"""
Test Runner for AI Strategy Engine

This script provides a comprehensive test runner for the AI strategy module,
including unit tests, integration tests, performance tests, and coverage reporting.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestRunner:
    """Manages test execution and reporting for AI strategy tests."""

    def __init__(self, verbose: bool = False):
        """
        Initialize test runner.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.test_dir = Path(__file__).parent
        self.results: dict[str, int] = {}

    def run_unit_tests(self) -> int:
        """
        Run unit tests.

        Returns:
            Exit code (0 for success)
        """
        print("\n" + "=" * 80)
        print("Running Unit Tests")
        print("=" * 80)

        cmd = [
            "pytest",
            "tests/test_ai_strategy.py",
            "tests/test_ai_strategy_extended.py",
            "-m",
            "not integration and not performance and not benchmark",
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--json-report",
            "--json-report-file=test_results_unit.json",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        self.results["unit"] = result.returncode
        return result.returncode

    def run_integration_tests(self) -> int:
        """
        Run integration tests.

        Returns:
            Exit code (0 for success)
        """
        print("\n" + "=" * 80)
        print("Running Integration Tests")
        print("=" * 80)

        cmd = [
            "pytest",
            "tests/test_ai_strategy_extended.py",
            "tests/test_ai_strategy_errors.py",
            "-m",
            "integration",
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--json-report",
            "--json-report-file=test_results_integration.json",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        self.results["integration"] = result.returncode
        return result.returncode

    def run_performance_tests(self) -> int:
        """
        Run performance tests.

        Returns:
            Exit code (0 for success)
        """
        print("\n" + "=" * 80)
        print("Running Performance Tests")
        print("=" * 80)

        cmd = [
            "pytest",
            "tests/test_ai_strategy_performance.py",
            "-m",
            "performance and not benchmark",
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--json-report",
            "--json-report-file=test_results_performance.json",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        self.results["performance"] = result.returncode
        return result.returncode

    def run_error_tests(self) -> int:
        """
        Run error handling tests.

        Returns:
            Exit code (0 for success)
        """
        print("\n" + "=" * 80)
        print("Running Error Handling Tests")
        print("=" * 80)

        cmd = [
            "pytest",
            "tests/test_ai_strategy_errors.py",
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--json-report",
            "--json-report-file=test_results_errors.json",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        self.results["errors"] = result.returncode
        return result.returncode

    def run_coverage(self) -> int:
        """
        Run tests with coverage analysis.

        Returns:
            Exit code (0 for success)
        """
        print("\n" + "=" * 80)
        print("Running Tests with Coverage Analysis")
        print("=" * 80)

        cmd = [
            "pytest",
            "tests/test_ai_strategy*.py",
            "-m",
            "not benchmark",
            "--cov=services/strategy_engine/src",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_ai_strategy",
            "--cov-report=json:coverage_ai_strategy.json",
            "--cov-branch",
            "-q",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        # Parse coverage results
        self._parse_coverage_results()

        self.results["coverage"] = result.returncode
        return result.returncode

    def run_benchmarks(self) -> int:
        """
        Run benchmark tests.

        Returns:
            Exit code (0 for success)
        """
        print("\n" + "=" * 80)
        print("Running Benchmark Tests")
        print("=" * 80)

        cmd = [
            "pytest",
            "tests/test_ai_strategy_performance.py",
            "-m",
            "benchmark",
            "--benchmark-only",
            "--benchmark-json=benchmark_results.json",
            "-v" if self.verbose else "-q",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        self.results["benchmarks"] = result.returncode
        return result.returncode

    def run_specific_test(self, test_name: str) -> int:
        """
        Run a specific test by name.

        Args:
            test_name: Name of the test to run

        Returns:
            Exit code (0 for success)
        """
        print("\n" + "=" * 80)
        print(f"Running Specific Test: {test_name}")
        print("=" * 80)

        cmd = ["pytest", "-k", test_name, "-v", "--tb=short"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)

        return result.returncode

    def run_all_tests(self) -> int:
        """
        Run all test suites.

        Returns:
            Exit code (0 for success, non-zero for any failures)
        """
        print("\n" + "=" * 80)
        print("AI STRATEGY TEST SUITE")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Run each test suite
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_error_tests()
        self.run_performance_tests()
        self.run_coverage()

        # Generate summary
        self._generate_summary()

        # Return overall result
        return 0 if all(code == 0 for code in self.results.values()) else 1

    def _parse_coverage_results(self):
        """Parse and display coverage results."""
        try:
            with open("coverage_ai_strategy.json", "r") as f:
                coverage_data = json.load(f)

            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
            print(f"\nTotal Coverage: {total_coverage:.2f}%")

            # Show file-level coverage
            files = coverage_data.get("files", {})
            print("\nFile Coverage:")
            for file_path, file_data in files.items():
                if (
                    "ai_strategy" in file_path
                    or "ai_models" in file_path
                    or "ai_integration" in file_path
                ):
                    coverage = file_data.get("summary", {}).get("percent_covered", 0)
                    print(f"  {Path(file_path).name}: {coverage:.2f}%")

        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Could not parse coverage results: {e}")

    def _generate_summary(self):
        """Generate and display test summary."""
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)

        total_passed = sum(1 for code in self.results.values() if code == 0)
        total_failed = sum(1 for code in self.results.values() if code != 0)

        print(f"\nTest Suites Run: {len(self.results)}")
        print(f"Passed: {total_passed}")
        print(f"Failed: {total_failed}")

        print("\nDetailed Results:")
        for suite, code in self.results.items():
            status = "✓ PASSED" if code == 0 else "✗ FAILED"
            print(f"  {suite.capitalize()}: {status}")

        if total_failed == 0:
            print("\n🎉 All tests passed successfully!")
        else:
            print(f"\n⚠️  {total_failed} test suite(s) failed.")

        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run AI Strategy Engine tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ai_tests.py --all              # Run all tests
  python run_ai_tests.py --unit             # Run only unit tests
  python run_ai_tests.py --coverage         # Run with coverage analysis
  python run_ai_tests.py --test test_cache  # Run specific test
  python run_ai_tests.py --benchmark        # Run benchmark tests
        """,
    )

    parser.add_argument("--all", action="store_true", help="Run all test suites")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests"
    )
    parser.add_argument(
        "--errors", action="store_true", help="Run error handling tests"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage analysis"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark tests")
    parser.add_argument("--test", type=str, help="Run specific test by name")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Create test runner
    runner = TestRunner(verbose=args.verbose)

    # Determine what to run
    exit_code = 0

    if args.all or (
        not any(
            [
                args.unit,
                args.integration,
                args.performance,
                args.errors,
                args.coverage,
                args.benchmark,
                args.test,
            ]
        )
    ):
        # Run all tests if --all or no specific option provided
        exit_code = runner.run_all_tests()

    else:
        # Run specific test suites
        if args.unit:
            exit_code |= runner.run_unit_tests()

        if args.integration:
            exit_code |= runner.run_integration_tests()

        if args.performance:
            exit_code |= runner.run_performance_tests()

        if args.errors:
            exit_code |= runner.run_error_tests()

        if args.coverage:
            exit_code |= runner.run_coverage()

        if args.benchmark:
            exit_code |= runner.run_benchmarks()

        if args.test:
            exit_code |= runner.run_specific_test(args.test)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
