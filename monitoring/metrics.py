import asyncio
import logging
import os

# Import shared models
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import asyncpg
import psutil
import redis.asyncio as redis
from prometheus_client import REGISTRY, Counter, Gauge, Histogram, start_http_server
from prometheus_client.core import CollectorRegistry

from shared.config import Config
from shared.models import PortfolioState

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from .gotify_client import GotifyClient  # noqa: E402


class TradingSystemMetrics:
    """Prometheus metrics for the trading system"""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY

        # Service Health Metrics
        self.service_up = Gauge(
            "service_up",
            "Service availability (1 = up, 0 = down)",
            ["service_name"],
            registry=self.registry,
        )

        self.service_uptime = Gauge(
            "service_uptime_seconds",
            "Service uptime in seconds",
            ["service_name"],
            registry=self.registry,
        )

        # API Metrics
        self.http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["service", "method", "endpoint", "status_code"],
            registry=self.registry,
        )

        self.http_request_duration = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["service", "method", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        # Trading Metrics
        self.trades_executed_total = Counter(
            "trades_executed_total",
            "Total number of trades executed",
            ["symbol", "side", "strategy"],
            registry=self.registry,
        )

        self.trade_execution_latency = Histogram(
            "trade_execution_latency_seconds",
            "Trade execution latency in seconds",
            ["symbol", "order_type"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        self.trade_pnl = Histogram(
            "trade_pnl_dollars",
            "Trade P&L in dollars",
            ["symbol", "strategy"],
            buckets=[-10000, -5000, -1000, -500, -100, 0, 100, 500, 1000, 5000, 10000],
            registry=self.registry,
        )

        self.winning_trades_total = Counter(
            "winning_trades_total",
            "Total number of winning trades",
            ["strategy"],
            registry=self.registry,
        )

        self.losing_trades_total = Counter(
            "losing_trades_total",
            "Total number of losing trades",
            ["strategy"],
            registry=self.registry,
        )

        # Portfolio Metrics
        self.portfolio_total_value = Gauge(
            "portfolio_total_value",
            "Total portfolio value in dollars",
            registry=self.registry,
        )

        self.portfolio_cash = Gauge(
            "portfolio_cash", "Cash available in dollars", registry=self.registry
        )

        self.portfolio_buying_power = Gauge(
            "portfolio_buying_power", "Buying power in dollars", registry=self.registry
        )

        self.portfolio_daily_pnl = Gauge(
            "portfolio_daily_pnl",
            "Daily portfolio P&L in dollars",
            registry=self.registry,
        )

        self.portfolio_cumulative_pnl = Gauge(
            "portfolio_cumulative_pnl",
            "Cumulative portfolio P&L in dollars",
            registry=self.registry,
        )

        self.portfolio_drawdown = Gauge(
            "portfolio_drawdown",
            "Current portfolio drawdown percentage",
            registry=self.registry,
        )

        self.portfolio_sharpe_ratio = Gauge(
            "portfolio_sharpe_ratio", "Portfolio Sharpe ratio", registry=self.registry
        )

        self.portfolio_sortino_ratio = Gauge(
            "portfolio_sortino_ratio", "Portfolio Sortino ratio", registry=self.registry
        )

        self.portfolio_beta = Gauge(
            "portfolio_beta",
            "Portfolio beta relative to market",
            registry=self.registry,
        )

        self.portfolio_var_95 = Gauge(
            "portfolio_var_95",
            "Portfolio Value at Risk (95% confidence)",
            registry=self.registry,
        )

        self.portfolio_expected_shortfall_95 = Gauge(
            "portfolio_expected_shortfall_95",
            "Portfolio Expected Shortfall (95% confidence)",
            registry=self.registry,
        )

        # Risk Metrics
        self.portfolio_risk_score = Gauge(
            "portfolio_risk_score",
            "Overall portfolio risk score (0-1)",
            registry=self.registry,
        )

        self.max_position_concentration = Gauge(
            "max_position_concentration",
            "Maximum position concentration percentage",
            registry=self.registry,
        )

        self.portfolio_correlation_risk = Gauge(
            "portfolio_correlation_risk",
            "Portfolio correlation risk score (0-1)",
            registry=self.registry,
        )

        self.risk_limit_violations_total = Counter(
            "risk_limit_violations_total",
            "Total number of risk limit violations",
            ["violation_type"],
            registry=self.registry,
        )

        self.portfolio_var_exceeded_count = Counter(
            "portfolio_var_exceeded_count",
            "Number of times VaR was exceeded",
            registry=self.registry,
        )

        # Market Data Metrics
        self.market_data_collection_total = Counter(
            "market_data_collection_total",
            "Total market data points collected",
            ["symbol", "timeframe", "source"],
            registry=self.registry,
        )

        self.market_data_collection_errors_total = Counter(
            "market_data_collection_errors_total",
            "Total market data collection errors",
            ["symbol", "error_type"],
            registry=self.registry,
        )

        self.market_data_last_update_timestamp = Gauge(
            "market_data_last_update_timestamp",
            "Timestamp of last market data update",
            ["symbol"],
            registry=self.registry,
        )

        self.data_validation_failures_total = Counter(
            "data_validation_failures_total",
            "Total data validation failures",
            ["symbol", "validation_type"],
            registry=self.registry,
        )

        # Strategy Metrics
        self.trading_signals_generated_total = Counter(
            "trading_signals_generated_total",
            "Total trading signals generated",
            ["symbol", "strategy", "signal_type"],
            registry=self.registry,
        )

        self.signal_confidence = Histogram(
            "signal_confidence",
            "Trading signal confidence scores",
            ["strategy"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        self.strategy_sharpe_ratio = Gauge(
            "strategy_sharpe_ratio",
            "Strategy Sharpe ratio",
            ["strategy"],
            registry=self.registry,
        )

        self.strategy_consecutive_losses = Gauge(
            "strategy_consecutive_losses",
            "Number of consecutive losses for strategy",
            ["strategy"],
            registry=self.registry,
        )

        # External API Metrics
        self.external_api_requests_total = Counter(
            "external_api_requests_total",
            "Total external API requests",
            ["api", "endpoint", "status"],
            registry=self.registry,
        )

        self.external_api_request_duration = Histogram(
            "external_api_request_duration_seconds",
            "External API request duration",
            ["api", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        self.alpaca_api_requests_total = Counter(
            "alpaca_api_requests_total",
            "Total Alpaca API requests",
            ["endpoint", "status"],
            registry=self.registry,
        )

        self.twelve_data_api_rate_limit_errors_total = Counter(
            "twelve_data_api_rate_limit_errors_total",
            "TwelveData API rate limit errors",
            registry=self.registry,
        )

        # Database Metrics
        self.database_connections_active = Gauge(
            "database_connections_active",
            "Number of active database connections",
            ["database"],
            registry=self.registry,
        )

        self.database_query_duration = Histogram(
            "database_query_duration_seconds",
            "Database query duration",
            ["query_type"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )

        self.database_errors_total = Counter(
            "database_errors_total",
            "Total database errors",
            ["error_type"],
            registry=self.registry,
        )

        self.database_size_bytes = Gauge(
            "database_size_bytes",
            "Database size in bytes",
            ["database"],
            registry=self.registry,
        )

        # Redis Metrics
        self.redis_operations_total = Counter(
            "redis_operations_total",
            "Total Redis operations",
            ["operation"],
            registry=self.registry,
        )

        self.redis_connected_clients = Gauge(
            "redis_connected_clients",
            "Number of connected Redis clients",
            registry=self.registry,
        )

        self.redis_memory_used_bytes = Gauge(
            "redis_memory_used_bytes",
            "Redis memory usage in bytes",
            registry=self.registry,
        )

        self.redis_memory_max_bytes = Gauge(
            "redis_memory_max_bytes",
            "Redis maximum memory in bytes",
            registry=self.registry,
        )

        self.redis_commands_processed_total = Gauge(
            "redis_commands_processed_total",
            "Total commands processed by Redis",
            registry=self.registry,
        )

        self.redis_instantaneous_ops_per_sec = Gauge(
            "redis_instantaneous_ops_per_sec",
            "Redis instantaneous operations per second",
            registry=self.registry,
        )

        self.redis_operation_duration = Histogram(
            "redis_operation_duration_seconds",
            "Redis operation duration",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
            registry=self.registry,
        )

        # System Resource Metrics
        self.system_cpu_usage = Gauge(
            "system_cpu_usage_percent",
            "System CPU usage percentage",
            registry=self.registry,
        )

        self.system_memory_usage = Gauge(
            "system_memory_usage_bytes",
            "System memory usage in bytes",
            registry=self.registry,
        )

        self.system_disk_usage = Gauge(
            "system_disk_usage_bytes",
            "System disk usage in bytes",
            ["mount_point"],
            registry=self.registry,
        )

        # Business Logic Metrics
        self.daily_trade_count = Gauge(
            "daily_trade_count",
            "Number of trades executed today",
            registry=self.registry,
        )

        self.daily_trade_limit = Gauge(
            "daily_trade_limit", "Daily trade limit", registry=self.registry
        )

        self.position_count = Gauge(
            "position_count", "Number of open positions", registry=self.registry
        )

        self.position_size_percentage = Gauge(
            "position_size_percentage",
            "Position size as percentage of portfolio",
            ["symbol"],
            registry=self.registry,
        )


class MetricsCollector:
    """Centralized metrics collection service"""

    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.metrics = TradingSystemMetrics()
        self.start_time = time.time()
        self._redis_client: Optional[redis.Redis] = None
        self._db_pool: Optional[asyncpg.Pool] = None
        self.collection_interval = 30  # seconds
        self._running = False

    async def startup(self):
        """Initialize metrics collector"""
        try:
            # Initialize Redis connection
            self._redis_client = redis.from_url(
                f"redis://{self.config.redis.host}:{self.config.redis.port}",
                password=self.config.redis.password,
                decode_responses=True,
            )

            # Initialize database connection pool
            self._db_pool = await asyncpg.create_pool(
                host=self.config.database.host,
                port=self.config.database.port,
                database=self.config.database.database,
                user=self.config.database.username,
                password=self.config.database.password,
                min_size=2,
                max_size=5,
            )

            # Start Prometheus HTTP server
            start_http_server(9090, registry=self.metrics.registry)

            self.logger.info("Metrics collector started on port 9090")

        except Exception as e:
            self.logger.error(f"Failed to start metrics collector: {e}")
            raise

    async def shutdown(self):
        """Cleanup metrics collector"""
        self._running = False

        if self._redis_client:
            await self._redis_client.close()

        if self._db_pool:
            await self._db_pool.close()

    async def start_collection(self):
        """Start periodic metrics collection"""
        self._running = True

        while self._running:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)

    async def collect_all_metrics(self):
        """Collect all system metrics"""
        await asyncio.gather(
            self.collect_service_metrics(),
            self.collect_portfolio_metrics(),
            self.collect_system_metrics(),
            self.collect_database_metrics(),
            self.collect_redis_metrics(),
            self.collect_trading_metrics(),
            self.collect_risk_metrics(),
            return_exceptions=True,
        )

    async def collect_service_metrics(self):
        """Collect service health metrics"""
        services = [
            ("data_collector", 9101),
            ("strategy_engine", 9102),
            ("risk_manager", 9103),
            ("trade_executor", 9104),
            ("scheduler", 9105),
        ]

        for service_name, port in services:
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://{service_name}:{port}/health", timeout=5.0
                    )

                    if response.status_code == 200:
                        self.metrics.service_up.labels(service_name=service_name).set(1)

                        health_data = response.json()
                        if "uptime" in health_data:
                            self.metrics.service_uptime.labels(
                                service_name=service_name
                            ).set(health_data["uptime"])
                    else:
                        self.metrics.service_up.labels(service_name=service_name).set(0)

            except Exception as e:
                self.logger.warning(
                    f"Failed to collect metrics for {service_name}: {e}"
                )
                self.metrics.service_up.labels(service_name=service_name).set(0)

    async def collect_portfolio_metrics(self):
        """Collect portfolio-related metrics"""
        try:
            if not self._db_pool:
                return

            async with self._db_pool.acquire() as conn:
                # Get latest portfolio state
                portfolio_row = await conn.fetchrow(
                    """
                    SELECT
                        total_value,
                        cash,
                        buying_power,
                        daily_pnl,
                        cumulative_pnl,
                        drawdown,
                        risk_score,
                        beta,
                        sharpe_ratio,
                        sortino_ratio
                    FROM portfolio_snapshots
                    ORDER BY timestamp DESC
                    LIMIT 1
                """
                )

                if portfolio_row:
                    self.metrics.portfolio_total_value.set(
                        portfolio_row["total_value"] or 0
                    )
                    self.metrics.portfolio_cash.set(portfolio_row["cash"] or 0)
                    self.metrics.portfolio_buying_power.set(
                        portfolio_row["buying_power"] or 0
                    )
                    self.metrics.portfolio_daily_pnl.set(
                        portfolio_row["daily_pnl"] or 0
                    )
                    self.metrics.portfolio_cumulative_pnl.set(
                        portfolio_row["cumulative_pnl"] or 0
                    )
                    self.metrics.portfolio_drawdown.set(portfolio_row["drawdown"] or 0)

                    if portfolio_row["beta"]:
                        self.metrics.portfolio_beta.set(portfolio_row["beta"])
                    if portfolio_row["sharpe_ratio"]:
                        self.metrics.portfolio_sharpe_ratio.set(
                            portfolio_row["sharpe_ratio"]
                        )
                    if portfolio_row["sortino_ratio"]:
                        self.metrics.portfolio_sortino_ratio.set(
                            portfolio_row["sortino_ratio"]
                        )

                # Get position metrics
                positions = await conn.fetch(
                    """
                    SELECT symbol, market_value, quantity
                    FROM positions
                    WHERE quantity != 0
                """
                )

                self.metrics.position_count.set(len(positions))

                # Calculate position concentrations
                total_value = portfolio_row["total_value"] if portfolio_row else 1
                max_concentration = 0

                for position in positions:
                    concentration = position["market_value"] / total_value
                    self.metrics.position_size_percentage.labels(
                        symbol=position["symbol"]
                    ).set(concentration)
                    max_concentration = max(max_concentration, concentration)

                self.metrics.max_position_concentration.set(max_concentration)

        except Exception as e:
            self.logger.error(f"Failed to collect portfolio metrics: {e}")

    async def collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics.system_cpu_usage.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self.metrics.system_memory_usage.set(memory.used)

            # Disk usage
            disk_usage = psutil.disk_usage("/")
            self.metrics.system_disk_usage.labels(mount_point="/").set(disk_usage.used)

            # Additional disk mounts if data directory is separate
            if os.path.exists("/app/data"):
                data_disk = psutil.disk_usage("/app/data")
                self.metrics.system_disk_usage.labels(mount_point="/app/data").set(
                    data_disk.used
                )

        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")

    async def collect_database_metrics(self):
        """Collect database metrics"""
        try:
            if not self._db_pool:
                return

            async with self._db_pool.acquire() as conn:
                # Active connections
                active_connections = await conn.fetchval(
                    "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'"
                )
                self.metrics.database_connections_active.labels(
                    database="postgres"
                ).set(active_connections)

                # Database size
                db_size = await conn.fetchval(
                    "SELECT pg_database_size(current_database())"
                )
                self.metrics.database_size_bytes.labels(database="postgres").set(
                    db_size or 0
                )

                # Recent query performance
                slow_queries = await conn.fetchval(
                    """
                    SELECT count(*)
                    FROM pg_stat_activity
                    WHERE state = 'active'
                    AND query_start < now() - interval '30 seconds'
                """
                )

                if slow_queries > 0:
                    self.logger.warning(f"Found {slow_queries} slow database queries")

        except Exception as e:
            self.logger.error(f"Failed to collect database metrics: {e}")

    async def collect_redis_metrics(self):
        """Collect Redis metrics"""
        try:
            if not self._redis_client:
                return

            # Redis info
            info = await self._redis_client.info()

            # Connected clients
            connected_clients = info.get("connected_clients", 0)
            self.metrics.redis_connected_clients.set(connected_clients)

            # Memory usage
            used_memory = info.get("used_memory", 0)
            max_memory = info.get("maxmemory", 0)
            self.metrics.redis_memory_used_bytes.set(used_memory)
            self.metrics.redis_memory_max_bytes.set(max_memory)

            # Commands processed
            total_commands = info.get("total_commands_processed", 0)
            self.metrics.redis_commands_processed_total.set(total_commands)

            # Operations per second
            instantaneous_ops = info.get("instantaneous_ops_per_sec", 0)
            self.metrics.redis_instantaneous_ops_per_sec.set(instantaneous_ops)

        except Exception as e:
            self.logger.error(f"Failed to collect Redis metrics: {e}")

    async def collect_trading_metrics(self):
        """Collect trading-specific metrics"""
        try:
            if not self._db_pool:
                return

            async with self._db_pool.acquire() as conn:
                # Daily trade count
                today = datetime.now(timezone.utc).date()
                daily_trades = await conn.fetchval(
                    """
                    SELECT count(*)
                    FROM trades
                    WHERE DATE(timestamp) = $1
                """,
                    today,
                )

                self.metrics.daily_trade_count.set(daily_trades)

                # Recent trade performance
                recent_trades = await conn.fetch(
                    """
                    SELECT
                        symbol,
                        strategy,
                        pnl,
                        EXTRACT(EPOCH FROM (executed_at - created_at)) as execution_time
                    FROM trades
                    WHERE timestamp >= NOW() - INTERVAL '1 hour'
                    ORDER BY timestamp DESC
                """
                )

                for trade in recent_trades:
                    # Record trade execution time
                    if trade["execution_time"]:
                        self.metrics.trade_execution_latency.labels(
                            symbol=trade["symbol"], order_type="market"
                        ).observe(trade["execution_time"])

                    # Record P&L
                    self.metrics.trade_pnl.labels(
                        symbol=trade["symbol"], strategy=trade["strategy"]
                    ).observe(trade["pnl"])

                    # Count winning/losing trades
                    if trade["pnl"] > 0:
                        self.metrics.winning_trades_total.labels(
                            strategy=trade["strategy"]
                        ).inc()
                    else:
                        self.metrics.losing_trades_total.labels(
                            strategy=trade["strategy"]
                        ).inc()

        except Exception as e:
            self.logger.error(f"Failed to collect trading metrics: {e}")

    async def collect_risk_metrics(self):
        """Collect risk management metrics"""
        try:
            if not self._db_pool:
                return

            async with self._db_pool.acquire() as conn:
                # Risk events
                risk_events = await conn.fetch(
                    """
                    SELECT event_type, severity
                    FROM risk_events
                    WHERE timestamp >= NOW() - INTERVAL '1 hour'
                """
                )

                for event in risk_events:
                    self.metrics.risk_limit_violations_total.labels(
                        violation_type=event["event_type"]
                    ).inc()

                # VaR breaches
                var_breaches = await conn.fetchval(
                    """
                    SELECT count(*)
                    FROM portfolio_snapshots
                    WHERE DATE(timestamp) = CURRENT_DATE
                    AND daily_pnl < var_95
                """
                )

                if var_breaches:
                    self.metrics.portfolio_var_exceeded_count.inc(var_breaches)

        except Exception as e:
            self.logger.error(f"Failed to collect risk metrics: {e}")

    # Metric recording methods for use by services
    def record_api_request(
        self,
        service: str,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
    ):
        """Record API request metrics"""
        self.metrics.http_requests_total.labels(
            service=service,
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        self.metrics.http_request_duration.labels(
            service=service, method=method, endpoint=endpoint
        ).observe(duration)

    def record_trade_execution(
        self,
        symbol: str,
        side: str,
        strategy: str,
        latency: float,
        pnl: Optional[float] = None,
    ):
        """Record trade execution metrics"""
        self.metrics.trades_executed_total.labels(
            symbol=symbol, side=side, strategy=strategy
        ).inc()

        self.metrics.trade_execution_latency.labels(
            symbol=symbol, order_type="market"
        ).observe(latency)

        if pnl is not None:
            self.metrics.trade_pnl.labels(symbol=symbol, strategy=strategy).observe(pnl)

    def record_signal_generation(
        self, symbol: str, strategy: str, signal_type: str, confidence: float
    ):
        """Record signal generation metrics"""
        self.metrics.trading_signals_generated_total.labels(
            symbol=symbol, strategy=strategy, signal_type=signal_type
        ).inc()

        self.metrics.signal_confidence.labels(strategy=strategy).observe(confidence)

    def record_market_data_collection(
        self,
        symbol: str,
        timeframe: str,
        source: str,
        success: bool,
        error_type: Optional[str] = None,
    ):
        """Record market data collection metrics"""
        if success:
            self.metrics.market_data_collection_total.labels(
                symbol=symbol, timeframe=timeframe, source=source
            ).inc()

            self.metrics.market_data_last_update_timestamp.labels(symbol=symbol).set(
                time.time()
            )
        else:
            self.metrics.market_data_collection_errors_total.labels(
                symbol=symbol, error_type=error_type or "unknown"
            ).inc()

    def record_external_api_request(
        self, api: str, endpoint: str, duration: float, success: bool
    ):
        """Record external API request metrics"""
        status = "success" if success else "error"

        self.metrics.external_api_requests_total.labels(
            api=api, endpoint=endpoint, status=status
        ).inc()

        self.metrics.external_api_request_duration.labels(
            api=api, endpoint=endpoint
        ).observe(duration)

    def record_database_operation(
        self,
        query_type: str,
        duration: float,
        success: bool,
        error_type: Optional[str] = None,
    ):
        """Record database operation metrics"""
        self.metrics.database_query_duration.labels(query_type=query_type).observe(
            duration
        )

        if not success and error_type:
            self.metrics.database_errors_total.labels(error_type=error_type).inc()

    def record_redis_operation(self, operation: str, duration: float, success: bool):
        """Record Redis operation metrics"""
        status = "success" if success else "error"

        self.metrics.redis_operations_total.labels(
            operation=operation, status=status
        ).inc()

        self.metrics.redis_operation_duration.labels(operation=operation).observe(
            duration
        )

    def update_portfolio_metrics(self, portfolio_state: PortfolioState):
        """Update portfolio metrics from portfolio state"""
        self.metrics.portfolio_total_value.set(float(portfolio_state.total_equity))
        self.metrics.portfolio_cash.set(float(portfolio_state.cash))
        self.metrics.portfolio_buying_power.set(float(portfolio_state.buying_power))

        # Calculate position metrics
        position_count = len(portfolio_state.positions)
        self.metrics.position_count.set(position_count)

        if portfolio_state.total_equity > 0:
            max_concentration = 0.0
            for position in portfolio_state.positions:
                concentration = float(position.market_value) / float(
                    portfolio_state.total_equity
                )
                self.metrics.position_size_percentage.labels(
                    symbol=position.symbol
                ).set(concentration)
                max_concentration = max(max_concentration, concentration)

            self.metrics.max_position_concentration.set(max_concentration)

    def record_risk_event(self, event_type: str, severity: str):
        """Record risk event"""
        self.metrics.risk_limit_violations_total.labels(violation_type=event_type).inc()

    @asynccontextmanager
    async def measure_duration(self, metric_name: str, labels: Dict[str, str]):
        """Context manager to measure operation duration"""
        start_time = time.time()
        try:
            yield
            success = True
        except Exception:
            success = False
            raise
        finally:
            duration = time.time() - start_time

            if metric_name == "api_request":
                self.record_api_request(
                    labels["service"],
                    labels["method"],
                    labels["endpoint"],
                    int(labels.get("status_code", 200)),
                    duration,
                )
            elif metric_name == "database_query":
                success_value = labels.get("success", True)
                if isinstance(success_value, str):
                    success = success_value.lower() == "true"
                else:
                    success = bool(success_value)
                self.record_database_operation(
                    labels["query_type"], duration, success, labels.get("error_type")
                )
            elif metric_name == "redis_operation":
                success_value = labels.get("success", True)
                if isinstance(success_value, str):
                    success = success_value.lower() == "true"
                else:
                    success = bool(success_value)
                self.record_redis_operation(labels["operation"], duration, success)


class AlertManager:
    """Custom alert manager for trading system"""

    def __init__(self, config: Config, metrics_collector: MetricsCollector):
        self.config = config
        self.metrics = metrics_collector
        self.logger = logging.getLogger(__name__)
        self.alert_rules: List[Dict[str, Any]] = []
        self.active_alerts: Dict[str, Dict] = {}
        self.gotify_client = None

        # Initialize Gotify client if configured
        if (
            hasattr(config, "notifications")
            and config.notifications.gotify_url
            and config.notifications.gotify_token
        ):
            self.gotify_client = GotifyClient(
                config.notifications.gotify_url, config.notifications.gotify_token
            )

    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add alert rule"""
        self.alert_rules.append(rule)

    async def check_alerts(self):
        """Check all alert rules"""
        for rule in self.alert_rules:
            try:
                await self._evaluate_alert_rule(rule)
            except Exception as e:
                self.logger.error(
                    f"Failed to evaluate alert rule {rule.get('name', 'unknown')}: {e}"
                )

    async def _evaluate_alert_rule(self, rule: Dict[str, Any]):
        """Evaluate a single alert rule"""
        rule_name = rule["name"]
        condition = rule["condition"]
        threshold = rule["threshold"]
        severity = rule.get("severity", "warning")

        # Get current metric value
        current_value = await self._get_metric_value(condition)

        if current_value is None:
            return

        # Check if alert should fire
        should_fire = False

        # Evaluate condition
        if condition == "greater_than":
            should_fire = current_value > threshold
        elif condition == "less_than":
            should_fire = current_value < threshold
        elif condition == "equals":
            should_fire = current_value == threshold
        elif condition == "not_equals":
            should_fire = current_value != threshold

        alert_key = f"{rule_name}_{condition}_{threshold}"

        if should_fire:
            if alert_key not in self.active_alerts:
                # Fire new alert
                alert_data = {
                    "rule_name": rule_name,
                    "condition": condition,
                    "threshold": threshold,
                    "current_value": current_value,
                    "severity": severity,
                    "fired_at": datetime.now(timezone.utc).isoformat(),
                }

                self.active_alerts[alert_key] = alert_data
                await self._fire_alert(alert_data)
        else:
            # Clear alert if it was active
            if alert_key in self.active_alerts:
                del self.active_alerts[alert_key]
                await self._clear_alert(alert_key)

    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric"""
        try:
            # This is a simplified implementation - in practice you'd query your metrics backend
            if hasattr(self.metrics, metric_name):
                metric = getattr(self.metrics, metric_name)
                if hasattr(metric, "_value"):
                    return float(metric._value._value)
                elif hasattr(metric, "get"):
                    return float(metric.get())
            return None
        except Exception as e:
            self.logger.error(f"Failed to get metric value for {metric_name}: {e}")
            return None

    async def _fire_alert(self, alert_data: Dict[str, Any]):
        """Fire an alert"""
        message = f"ALERT: {alert_data['rule_name']} - Current: {alert_data['current_value']}, Threshold: {alert_data['threshold']}"

        if self.gotify_client:
            try:
                await self.gotify_client.send_critical_alert(
                    alert_data["rule_name"], message, {"alert_data": alert_data}
                )
            except Exception as e:
                self.logger.error(f"Failed to send alert notification: {e}")

        self.logger.warning(f"Alert fired: {message}")

    async def _clear_alert(self, alert_key: str):
        """Clear an alert"""
        self.logger.info(f"Alert cleared: {alert_key}")
