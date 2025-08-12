# Makefile for AI Strategy Engine Tests
# Run with: make <target>

.PHONY: help test test-all test-unit test-integration test-performance test-errors \
        test-coverage test-benchmark test-watch clean install-deps lint type-check \
        format test-docker test-parallel report

# Variables
PYTHON := python3
PIP := pip3
PYTEST := pytest
PROJECT_DIR := ..
TEST_DIR := .
COVERAGE_DIR := coverage_ai_strategy
VENV := ../venv

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
help:
	@echo "$(BLUE)AI Strategy Engine Test Suite$(NC)"
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"
	@echo ""
	@echo "$(GREEN)Available targets:$(NC)"
	@echo "  $(BLUE)test$(NC)           - Run all tests (default)"
	@echo "  $(BLUE)test-unit$(NC)      - Run unit tests only"
	@echo "  $(BLUE)test-integration$(NC) - Run integration tests"
	@echo "  $(BLUE)test-performance$(NC) - Run performance tests"
	@echo "  $(BLUE)test-errors$(NC)    - Run error handling tests"
	@echo "  $(BLUE)test-coverage$(NC)  - Run tests with coverage report"
	@echo "  $(BLUE)test-benchmark$(NC) - Run benchmark tests"
	@echo "  $(BLUE)test-parallel$(NC)  - Run tests in parallel"
	@echo "  $(BLUE)test-watch$(NC)     - Run tests in watch mode"
	@echo "  $(BLUE)test-docker$(NC)    - Run tests in Docker container"
	@echo ""
	@echo "$(GREEN)Code quality:$(NC)"
	@echo "  $(BLUE)lint$(NC)           - Run linting checks"
	@echo "  $(BLUE)type-check$(NC)     - Run type checking with mypy"
	@echo "  $(BLUE)format$(NC)         - Format code with black"
	@echo ""
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  $(BLUE)clean$(NC)          - Clean test artifacts"
	@echo "  $(BLUE)install-deps$(NC)   - Install test dependencies"
	@echo "  $(BLUE)report$(NC)         - Generate HTML test report"
	@echo ""
	@echo "$(YELLOW)━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━$(NC)"

# Install test dependencies
install-deps:
	@echo "$(YELLOW)Installing test dependencies...$(NC)"
	$(PIP) install -r requirements-test.txt
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

# Run all tests
test: clean
	@echo "$(YELLOW)Running all AI Strategy tests...$(NC)"
	$(PYTHON) run_ai_tests.py --all

test-all: test

# Run unit tests
test-unit:
	@echo "$(YELLOW)Running unit tests...$(NC)"
	$(PYTEST) test_ai_strategy.py test_ai_strategy_extended.py \
		-m "not integration and not performance and not benchmark" \
		-v --tb=short \
		--json-report --json-report-file=results_unit.json

# Run integration tests
test-integration:
	@echo "$(YELLOW)Running integration tests...$(NC)"
	$(PYTEST) test_ai_strategy_extended.py test_ai_strategy_errors.py \
		-m "integration" \
		-v --tb=short \
		--json-report --json-report-file=results_integration.json

# Run performance tests
test-performance:
	@echo "$(YELLOW)Running performance tests...$(NC)"
	$(PYTEST) test_ai_strategy_performance.py \
		-m "performance and not benchmark" \
		-v --tb=short \
		--json-report --json-report-file=results_performance.json

# Run error handling tests
test-errors:
	@echo "$(YELLOW)Running error handling tests...$(NC)"
	$(PYTEST) test_ai_strategy_errors.py \
		-v --tb=short \
		--json-report --json-report-file=results_errors.json

# Run tests with coverage
test-coverage:
	@echo "$(YELLOW)Running tests with coverage analysis...$(NC)"
	$(PYTEST) test_ai_strategy*.py \
		-m "not benchmark" \
		--cov=../services/strategy_engine/src \
		--cov-report=term-missing \
		--cov-report=html:$(COVERAGE_DIR) \
		--cov-report=json:coverage.json \
		--cov-branch \
		--cov-fail-under=70
	@echo "$(GREEN)✓ Coverage report generated in $(COVERAGE_DIR)/index.html$(NC)"

# Run benchmark tests
test-benchmark:
	@echo "$(YELLOW)Running benchmark tests...$(NC)"
	$(PYTEST) test_ai_strategy_performance.py \
		-m "benchmark" \
		--benchmark-only \
		--benchmark-json=benchmark_results.json \
		--benchmark-autosave \
		--benchmark-compare \
		--benchmark-histogram

# Run tests in parallel
test-parallel:
	@echo "$(YELLOW)Running tests in parallel...$(NC)"
	$(PYTEST) test_ai_strategy*.py \
		-m "not benchmark" \
		-n auto \
		--dist loadscope \
		-v

# Run tests in watch mode
test-watch:
	@echo "$(YELLOW)Starting test watch mode...$(NC)"
	$(PYTEST) test_ai_strategy*.py \
		-m "not benchmark and not slow" \
		--watch

# Run specific test
test-specific:
	@echo "$(YELLOW)Running specific test: $(TEST)$(NC)"
	$(PYTEST) -k "$(TEST)" -v --tb=short

# Linting
lint:
	@echo "$(YELLOW)Running linting checks...$(NC)"
	@echo "$(BLUE)Running flake8...$(NC)"
	flake8 ../services/strategy_engine/src/ai_*.py --max-line-length=120 --ignore=E203,W503
	@echo "$(BLUE)Running pylint...$(NC)"
	pylint ../services/strategy_engine/src/ai_*.py --max-line-length=120 --disable=C0103,C0114,C0115,C0116
	@echo "$(GREEN)✓ Linting complete$(NC)"

# Type checking
type-check:
	@echo "$(YELLOW)Running type checking...$(NC)"
	mypy ../services/strategy_engine/src/ai_*.py \
		--ignore-missing-imports \
		--no-strict-optional \
		--warn-return-any \
		--warn-unused-ignores
	@echo "$(GREEN)✓ Type checking complete$(NC)"

# Format code
format:
	@echo "$(YELLOW)Formatting code...$(NC)"
	black ../services/strategy_engine/src/ai_*.py test_ai_*.py
	isort ../services/strategy_engine/src/ai_*.py test_ai_*.py
	@echo "$(GREEN)✓ Code formatted$(NC)"

# Generate HTML report
report:
	@echo "$(YELLOW)Generating HTML test report...$(NC)"
	$(PYTEST) test_ai_strategy*.py \
		-m "not benchmark" \
		--html=test_report.html \
		--self-contained-html \
		-v
	@echo "$(GREEN)✓ Report generated: test_report.html$(NC)"

# Run tests in Docker
test-docker:
	@echo "$(YELLOW)Building Docker test image...$(NC)"
	docker build -f Dockerfile.test -t ai-strategy-tests ..
	@echo "$(YELLOW)Running tests in Docker...$(NC)"
	docker run --rm \
		-v $(PWD):/tests \
		-e ANTHROPIC_API_KEY=test_key \
		-e DATABASE_URL=postgresql://test:test@localhost/test \
		-e REDIS_URL=redis://localhost:6379 \
		ai-strategy-tests

# Clean test artifacts
clean:
	@echo "$(YELLOW)Cleaning test artifacts...$(NC)"
	rm -rf $(COVERAGE_DIR)
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf ../**/__pycache__
	rm -rf ../**/.pytest_cache
	rm -f .coverage
	rm -f coverage.json
	rm -f test_report.html
	rm -f results_*.json
	rm -f benchmark_results.json
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "$(GREEN)✓ Cleaned$(NC)"

# Quick test (fast subset for development)
test-quick:
	@echo "$(YELLOW)Running quick tests...$(NC)"
	$(PYTEST) test_ai_strategy.py::TestAnthropicClient \
		test_ai_strategy.py::TestDataContextBuilder \
		-v --tb=short -x

# Test with warnings
test-warnings:
	@echo "$(YELLOW)Running tests with warnings enabled...$(NC)"
	$(PYTEST) test_ai_strategy*.py \
		-m "not benchmark" \
		-W default \
		--strict-markers \
		-v

# Continuous Integration target
ci: clean lint type-check test-coverage
	@echo "$(GREEN)✓ CI pipeline complete$(NC)"

# Generate test statistics
stats:
	@echo "$(YELLOW)Generating test statistics...$(NC)"
	@echo ""
	@echo "$(BLUE)Test Files:$(NC)"
	@ls -la test_ai_*.py | wc -l
	@echo ""
	@echo "$(BLUE)Test Functions:$(NC)"
	@grep -h "def test_" test_ai_*.py | wc -l
	@echo ""
	@echo "$(BLUE)Test Classes:$(NC)"
	@grep -h "class Test" test_ai_*.py | wc -l
	@echo ""
	@echo "$(BLUE)Code Coverage:$(NC)"
	@if [ -f coverage.json ]; then \
		python -c "import json; data=json.load(open('coverage.json')); print(f'{data[\"totals\"][\"percent_covered\"]:.2f}%')"; \
	else \
		echo "No coverage data available. Run 'make test-coverage' first."; \
	fi

# Check test health
health-check:
	@echo "$(YELLOW)Checking test suite health...$(NC)"
	@echo ""
	@echo "$(BLUE)Checking for broken imports...$(NC)"
	$(PYTHON) -c "import test_ai_strategy; import test_ai_strategy_extended; import test_ai_strategy_errors; import test_ai_strategy_performance"
	@echo "$(GREEN)✓ All test modules importable$(NC)"
	@echo ""
	@echo "$(BLUE)Checking for test discovery...$(NC)"
	$(PYTEST) --collect-only -q test_ai_*.py
	@echo "$(GREEN)✓ Test discovery successful$(NC)"

# Profile memory usage
profile-memory:
	@echo "$(YELLOW)Profiling memory usage...$(NC)"
	mprof run $(PYTEST) test_ai_strategy_performance.py::TestMemoryUsage -v
	mprof plot

# Security check
security-check:
	@echo "$(YELLOW)Running security checks...$(NC)"
	bandit -r ../services/strategy_engine/src/ai_*.py -f json -o security_report.json
	@echo "$(GREEN)✓ Security check complete. Report: security_report.json$(NC)"

.DEFAULT_GOAL := help
