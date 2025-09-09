"""
HTTP server for the data collection service.

This module provides a lightweight HTTP server that exposes health check
endpoints and basic service information for integration with the scheduler
and other services in the trading system.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, cast

from aiohttp import web
from aiohttp.web_request import Request
from aiohttp.web_response import Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest

logger = logging.getLogger(__name__)


class DataCollectorHTTPServer:
    """HTTP server for data collector service endpoints."""

    def __init__(
        self,
        data_service: Optional[Any] = None,
        port: int = 9101,
        host: str = "0.0.0.0",
    ):
        """
        Initialize the HTTP server.

        Args:
            data_service: Reference to the DataCollectionService instance
            port: Port to bind the server to
            host: Host to bind the server to
        """
        self.data_service = data_service
        self.port = port
        self.host = host
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.logger = logging.getLogger(__name__)

        # Initialize Prometheus metrics
        self._init_prometheus_metrics()

    async def create_app(self) -> web.Application:
        """Create and configure the web application."""
        app = web.Application()

        # Add routes
        app.router.add_get("/", self.index)
        app.router.add_get("/health", self.health_check)
        app.router.add_get("/status", self.status)
        app.router.add_get("/metrics", self.prometheus_metrics)
        app.router.add_get("/metrics/json", self.metrics)
        app.router.add_get("/info", self.info)

        # Add data collection endpoints
        app.router.add_post("/market-data/update", self.update_market_data)
        app.router.add_post("/finviz/scan", self.trigger_finviz_scan)
        app.router.add_post("/tickers/cleanup", self.cleanup_expired_tickers)
        app.router.add_get("/tickers/statistics", self.get_ticker_statistics)

        # Add data access endpoints for other services
        app.router.add_get("/market-data/historical/{symbol}", self.get_historical_data)
        app.router.add_get("/market-data/latest/{symbol}", self.get_latest_data)
        app.router.add_get(
            "/market-data/volatility/{symbol}", self.get_symbol_volatility
        )
        app.router.add_get(
            "/market-data/correlation/{symbol1}/{symbol2}", self.get_symbol_correlation
        )
        app.router.add_get("/market-data/atr/{symbol}", self.get_atr)

        # Add TwelveData API relay endpoints
        app.router.add_get("/api/twelvedata/quote/{symbol}", self.get_real_time_quote)
        app.router.add_get("/api/twelvedata/time-series/{symbol}", self.get_time_series)
        app.router.add_post("/api/twelvedata/batch-quotes", self.get_batch_quotes)
        app.router.add_get("/api/twelvedata/search/{query}", self.search_symbols)
        app.router.add_get(
            "/api/twelvedata/technical/{indicator}/{symbol}",
            self.get_technical_indicator,
        )

        # Add middleware for CORS and error handling
        app.middlewares.append(self.cors_handler)
        app.middlewares.append(self.error_handler)

        return app

    async def start(self) -> None:
        """Start the HTTP server."""
        try:
            self.app = await self.create_app()
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            self.site = web.TCPSite(self.runner, self.host, self.port)
            await self.site.start()

            self.logger.info(f"HTTP server started on {self.host}:{self.port}")

        except Exception as e:
            self.logger.error(f"Failed to start HTTP server: {e}")
            raise

    async def stop(self) -> None:
        """Stop the HTTP server."""
        try:
            if self.site:
                await self.site.stop()

            if self.runner:
                await self.runner.cleanup()

            self.logger.info("HTTP server stopped")

        except Exception as e:
            self.logger.error(f"Error stopping HTTP server: {e}")

    @web.middleware
    async def cors_handler(self, request: Request, handler) -> Response:
        """CORS middleware."""
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @web.middleware
    async def error_handler(self, request: Request, handler) -> Response:
        """Error handling middleware."""
        try:
            return await handler(request)
        except Exception as e:
            self.logger.error(f"HTTP request error: {e}")
            return web.json_response(
                {
                    "error": "Internal server error",
                    "message": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=500,
            )

    async def index(self, request: Request) -> Response:
        """Root endpoint with basic service information."""
        return web.json_response(
            {
                "service": "data_collector",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "endpoints": {
                    "health": "/health",
                    "status": "/status",
                    "metrics": "/metrics",
                    "metrics_json": "/metrics/json",
                    "info": "/info",
                },
            }
        )

    async def health_check(self, request: Request) -> Response:
        """Health check endpoint for service monitoring."""
        try:
            if not self.data_service:
                return web.json_response(
                    {
                        "status": "unhealthy",
                        "message": "Data service not available",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    status=503,
                )

            # Get health information from data service
            health_info = await self._get_health_info()

            # Determine HTTP status code based on health
            status_code = 200
            if health_info["status"] == "unhealthy":
                status_code = 503
            elif health_info["status"] == "degraded":
                status_code = 200  # Still responsive but degraded

            return web.json_response(health_info, status=status_code)

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return web.json_response(
                {
                    "status": "error",
                    "message": f"Health check failed: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                status=500,
            )

    async def status(self, request: Request) -> Response:
        """Detailed status endpoint."""
        try:
            if not self.data_service:
                return web.json_response(
                    {"error": "Data service not available"}, status=503
                )

            # Get service status
            status = await self.data_service.get_service_status()

            return web.json_response(
                {
                    "service": "data_collector",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": status,
                }
            )

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return web.json_response(
                {"error": f"Status check failed: {str(e)}"}, status=500
            )

    async def prometheus_metrics(self, request: Request) -> Response:
        """Prometheus-style metrics endpoint."""
        try:
            if not self.data_service:
                return web.Response(
                    text="# Data service not available\n",
                    content_type="text/plain; version=0.0.4",
                    status=503,
                )

            # Update Prometheus metrics with current values
            await self._update_prometheus_metrics()

            # Generate Prometheus format
            metrics_output = generate_latest()

            return web.Response(
                text=metrics_output.decode("utf-8"),
                content_type="text/plain; version=0.0.4",
            )

        except Exception as e:
            self.logger.error(f"Prometheus metrics collection failed: {e}")
            return web.Response(
                text=f"# Error collecting metrics: {str(e)}\n",
                content_type="text/plain; version=0.0.4",
                status=500,
            )

    async def metrics(self, request: Request) -> Response:
        """JSON metrics endpoint for monitoring systems."""
        try:
            if not self.data_service:
                return web.json_response(
                    {"error": "Data service not available"}, status=503
                )

            # Get metrics from data service
            status = await self.data_service.get_service_status()

            metrics = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "service": "data_collector",
                "metrics": {
                    "active_tickers_count": status.get("active_tickers_count", 0),
                    "scheduled_jobs_count": len(status.get("scheduler_jobs", [])),
                    "is_running": status.get("is_running", False),
                    "statistics": status.get("statistics", {}),
                },
            }

            return web.json_response(metrics)

        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            return web.json_response(
                {"error": f"Metrics collection failed: {str(e)}"}, status=500
            )

    async def info(self, request: Request) -> Response:
        """Service information endpoint."""
        try:
            if not self.data_service:
                config_info = {"error": "Data service not available"}
            else:
                # Get configuration information (sanitized)
                config = (
                    self.data_service.config.dict()
                    if hasattr(self.data_service, "config")
                    else {}
                )

                # Remove sensitive information
                config_info = {
                    k: v
                    for k, v in config.items()
                    if k not in ["api_key", "password", "secret"]
                }

            return web.json_response(
                {
                    "service": "data_collector",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "configuration": config_info,
                    "server_info": {
                        "host": self.host,
                        "port": self.port,
                        "started": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )

        except Exception as e:
            self.logger.error(f"Info request failed: {e}")
            return web.json_response(
                {"error": f"Info request failed: {str(e)}"}, status=500
            )

    async def _get_health_info(self) -> Dict[str, Any]:
        """Get comprehensive health information from the data service."""
        health_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "data_collector",
            "status": "unknown",
            "components": {},
            "errors": [],
        }

        try:
            # Get service status
            service_status = await self.data_service.get_service_status()

            # Check if service is running
            is_running = service_status.get("is_running", False)
            components_dict = cast(Dict[str, Any], health_info["components"])
            components_dict["service"] = {
                "status": "healthy" if is_running else "unhealthy",
                "running": is_running,
                "active_tickers": service_status.get("active_tickers_count", 0),
            }

            # Check individual components if available
            if (
                hasattr(self.data_service, "finviz_screener")
                and self.data_service.finviz_screener
            ):
                try:
                    finviz_healthy = (
                        await self.data_service.finviz_screener.validate_connection()
                    )
                    components_dict = cast(Dict[str, Any], health_info["components"])
                    components_dict["finviz"] = {
                        "status": "healthy" if finviz_healthy else "degraded"
                    }
                    if not finviz_healthy:
                        errors_list = cast(List[str], health_info["errors"])
                        errors_list.append("FinViz connection issues")
                except Exception:
                    components_dict = cast(Dict[str, Any], health_info["components"])
                    components_dict["finviz"] = {"status": "unknown"}

            if (
                hasattr(self.data_service, "twelvedata_client")
                and self.data_service.twelvedata_client
            ):
                try:
                    twelvedata_healthy = (
                        await self.data_service.twelvedata_client.test_connection()
                    )
                    components_dict = cast(Dict[str, Any], health_info["components"])
                    components_dict["twelvedata"] = {
                        "status": "healthy" if twelvedata_healthy else "degraded"
                    }
                    if not twelvedata_healthy:
                        errors_list = cast(List[str], health_info["errors"])
                        errors_list.append("TwelveData connection issues")
                except Exception:
                    components_dict = cast(Dict[str, Any], health_info["components"])
                    components_dict["twelvedata"] = {"status": "unknown"}

            if (
                hasattr(self.data_service, "redis_client")
                and self.data_service.redis_client
            ):
                try:
                    redis_health = await self.data_service.redis_client.health_check()
                    components_dict = cast(Dict[str, Any], health_info["components"])
                    components_dict["redis"] = {
                        "status": (
                            "healthy"
                            if redis_health.get("connected", False)
                            else "unhealthy"
                        ),
                        "connected": redis_health.get("connected", False),
                    }
                    if not redis_health.get("connected", False):
                        errors_list = cast(List[str], health_info["errors"])
                        errors_list.append("Redis connection failed")
                except Exception:
                    components_dict = cast(Dict[str, Any], health_info["components"])
                    components_dict["redis"] = {"status": "unknown"}
                    errors_list = cast(List[str], health_info["errors"])
                    errors_list.append("Redis health check failed")

            # Determine overall health status
            components_dict = cast(Dict[str, Any], health_info["components"])
            component_statuses = [
                comp.get("status")
                for comp in components_dict.values()
                if isinstance(comp, dict) and "status" in comp
            ]

            if all(status == "healthy" for status in component_statuses):
                health_info["status"] = "healthy"
            elif any(status == "healthy" for status in component_statuses):
                health_info["status"] = "degraded"
            else:
                health_info["status"] = "unhealthy"

        except Exception as e:
            health_info["status"] = "error"
            errors_list = cast(List[str], health_info["errors"])
            errors_list.append(f"Health check failed: {str(e)}")

        return health_info

    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Service status metrics
        self.active_tickers_gauge = Gauge(
            "data_collector_active_tickers_total",
            "Number of active tickers being tracked",
        )

        self.scheduled_jobs_gauge = Gauge(
            "data_collector_scheduled_jobs_total", "Number of scheduled jobs"
        )

        self.service_running_gauge = Gauge(
            "data_collector_service_running",
            "Whether the data collection service is running (1=running, 0=stopped)",
        )

        # Statistics metrics
        self.screener_runs_counter = Counter(
            "data_collector_screener_runs_total", "Total number of screener runs"
        )

        self.data_updates_counter = Counter(
            "data_collector_data_updates_total", "Total number of data updates"
        )

        self.records_saved_counter = Counter(
            "data_collector_records_saved_total", "Total number of records saved"
        )

        self.errors_counter = Counter(
            "data_collector_errors_total", "Total number of errors encountered"
        )

        # Component health metrics
        self.component_health_gauge = Gauge(
            "data_collector_component_health",
            "Health status of components (1=healthy, 0=unhealthy)",
            ["component"],
        )

        # HTTP request metrics
        self.http_requests_counter = Counter(
            "data_collector_http_requests_total",
            "Total HTTP requests",
            ["endpoint", "method", "status"],
        )

        self.http_request_duration = Histogram(
            "data_collector_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["endpoint", "method"],
        )

    async def _update_prometheus_metrics(self) -> None:
        """Update Prometheus metrics with current values."""
        if not self.data_service:
            return

        try:
            # Get service status
            status = await self.data_service.get_service_status()
            statistics = status.get("statistics", {})

            # Update basic metrics
            self.active_tickers_gauge.set(status.get("active_tickers_count", 0))
            self.scheduled_jobs_gauge.set(len(status.get("scheduler_jobs", [])))
            self.service_running_gauge.set(1 if status.get("is_running", False) else 0)

            # Update statistics (these are cumulative, so we set them directly)
            screener_runs = statistics.get("screener_runs", 0)
            data_updates = statistics.get("data_updates", 0)
            records_saved = statistics.get("total_records_saved", 0)
            errors = statistics.get("errors", 0)

            # Set counter values (note: this is not ideal for counters, but needed for compatibility)
            self.screener_runs_counter._value._value = screener_runs
            self.data_updates_counter._value._value = data_updates
            self.records_saved_counter._value._value = records_saved
            self.errors_counter._value._value = errors

            # Update component health
            health_info = await self._get_health_info()
            components = health_info.get("components", {})

            for component_name, component_info in components.items():
                if isinstance(component_info, dict) and "status" in component_info:
                    health_value = 1 if component_info["status"] == "healthy" else 0
                    self.component_health_gauge.labels(component=component_name).set(
                        health_value
                    )

        except Exception as e:
            self.logger.error(f"Failed to update Prometheus metrics: {e}")

    async def update_market_data(self, request: web.Request) -> web.Response:
        """
        Trigger market data update.

        Accepts JSON body with optional 'timeframe' parameter.
        """
        try:
            data = await request.json() if request.body_exists else {}
            timeframe_str = data.get("timeframe", "5m")

            # Map timeframe string to TimeFrame enum
            from shared.models import TimeFrame

            timeframe_map = {
                "1m": TimeFrame.ONE_MINUTE,
                "5m": TimeFrame.FIVE_MINUTES,
                "15m": TimeFrame.FIFTEEN_MINUTES,
                "30m": TimeFrame.THIRTY_MINUTES,
                "1h": TimeFrame.ONE_HOUR,
                "1d": TimeFrame.ONE_DAY,
            }

            timeframe = timeframe_map.get(timeframe_str, TimeFrame.FIVE_MINUTES)

            # Trigger the update
            if hasattr(self.data_service, "_update_price_data"):
                await self.data_service._update_price_data(timeframe)

                return web.json_response(
                    {
                        "status": "success",
                        "message": f"Market data update triggered for {timeframe_str}",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            else:
                return web.json_response(
                    {
                        "status": "error",
                        "message": "Data collection service not available",
                    },
                    status=503,
                )

        except Exception as e:
            self.logger.error(f"Error triggering market data update: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def trigger_finviz_scan(self, request: web.Request) -> web.Response:
        """
        Trigger FinViz screener scan.
        """
        try:
            # Trigger the scan
            if hasattr(self.data_service, "_run_finviz_scan"):
                await self.data_service._run_finviz_scan()

                return web.json_response(
                    {
                        "status": "success",
                        "message": "FinViz scan triggered",
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            else:
                return web.json_response(
                    {"status": "error", "message": "FinViz screener not available"},
                    status=503,
                )

        except Exception as e:
            self.logger.error(f"Error triggering FinViz scan: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def cleanup_expired_tickers(self, request: web.Request) -> web.Response:
        """
        Cleanup tickers that haven't been seen from screener for specified hours.

        Query parameters:
        - hours: Number of hours after which to expire tickers (default: 1.0)
        """
        try:
            # Get expiry hours from query parameters
            expiry_hours = float(request.query.get("hours", 1.0))

            if expiry_hours < 0.1 or expiry_hours > 24:
                return web.json_response(
                    {
                        "status": "error",
                        "message": "hours parameter must be between 0.1 and 24",
                    },
                    status=400,
                )

            # Trigger the cleanup
            if hasattr(self.data_service, "cleanup_expired_tickers"):
                removed_tickers = await self.data_service.cleanup_expired_tickers(
                    expiry_hours
                )

                return web.json_response(
                    {
                        "status": "success",
                        "message": "Ticker cleanup completed",
                        "removed_tickers": removed_tickers,
                        "removed_count": len(removed_tickers),
                        "expiry_hours": expiry_hours,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            else:
                return web.json_response(
                    {"status": "error", "message": "Ticker cleanup not available"},
                    status=503,
                )

        except ValueError:
            return web.json_response(
                {
                    "status": "error",
                    "message": "Invalid hours parameter - must be a number",
                },
                status=400,
            )
        except Exception as e:
            self.logger.error(f"Error during ticker cleanup: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def get_ticker_statistics(self, request: web.Request) -> web.Response:
        """
        Get ticker management statistics.
        """
        try:
            if hasattr(self.data_service, "get_ticker_statistics"):
                stats = await self.data_service.get_ticker_statistics()

                return web.json_response(
                    {
                        "status": "success",
                        "data": stats,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            else:
                return web.json_response(
                    {"status": "error", "message": "Ticker statistics not available"},
                    status=503,
                )

        except Exception as e:
            self.logger.error(f"Error getting ticker statistics: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def get_historical_data(self, request: web.Request) -> web.Response:
        """
        Get historical market data for a symbol.

        Query parameters:
        - days: Number of days of data (default: 30)
        - timeframe: Data timeframe (default: 1d)
        """
        try:
            symbol = request.match_info["symbol"]
            days = int(request.query.get("days", 30))
            timeframe_str = request.query.get("timeframe", "1d")

            # Map timeframe string to TimeFrame enum
            from shared.models import TimeFrame

            timeframe_map = {
                "1m": TimeFrame.ONE_MINUTE,
                "5m": TimeFrame.FIVE_MINUTES,
                "15m": TimeFrame.FIFTEEN_MINUTES,
                "30m": TimeFrame.THIRTY_MINUTES,
                "1h": TimeFrame.ONE_HOUR,
                "1d": TimeFrame.ONE_DAY,
            }

            timeframe = timeframe_map.get(timeframe_str, TimeFrame.ONE_DAY)

            if not hasattr(self.data_service, "data_store"):
                return web.json_response(
                    {"status": "error", "message": "Data store not available"},
                    status=503,
                )

            # Calculate date range
            from datetime import datetime, timedelta

            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            # Get data from store
            df = await self.data_service.data_store.load_market_data(
                ticker=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or len(df) == 0:
                return web.json_response(
                    {"status": "success", "data": [], "symbol": symbol, "days": days}
                )

            # Convert DataFrame to list of dictionaries
            data = []
            for row in df.iter_rows(named=True):
                data.append(
                    {
                        "timestamp": (
                            row["timestamp"].isoformat() if row["timestamp"] else None
                        ),
                        "open": float(row["open"]) if row["open"] is not None else None,
                        "high": float(row["high"]) if row["high"] is not None else None,
                        "low": float(row["low"]) if row["low"] is not None else None,
                        "close": (
                            float(row["close"]) if row["close"] is not None else None
                        ),
                        "volume": (
                            int(row["volume"]) if row["volume"] is not None else None
                        ),
                    }
                )

            return web.json_response(
                {
                    "status": "success",
                    "data": data,
                    "symbol": symbol,
                    "days": days,
                    "timeframe": timeframe_str,
                }
            )

        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def get_latest_data(self, request: web.Request) -> web.Response:
        """
        Get latest market data for a symbol.

        Query parameters:
        - limit: Number of latest records (default: 1)
        - timeframe: Data timeframe (default: 1d)
        """
        try:
            symbol = request.match_info["symbol"]
            limit = int(request.query.get("limit", 1))
            timeframe_str = request.query.get("timeframe", "1d")

            # Map timeframe string to TimeFrame enum
            from shared.models import TimeFrame

            timeframe_map = {
                "1m": TimeFrame.ONE_MINUTE,
                "5m": TimeFrame.FIVE_MINUTES,
                "15m": TimeFrame.FIFTEEN_MINUTES,
                "30m": TimeFrame.THIRTY_MINUTES,
                "1h": TimeFrame.ONE_HOUR,
                "1d": TimeFrame.ONE_DAY,
            }

            timeframe = timeframe_map.get(timeframe_str, TimeFrame.ONE_DAY)

            if not hasattr(self.data_service, "data_store"):
                return web.json_response(
                    {"status": "error", "message": "Data store not available"},
                    status=503,
                )

            # Get latest data
            df = await self.data_service.data_store.get_latest_data(
                ticker=symbol, timeframe=timeframe, limit=limit
            )

            if df is None or len(df) == 0:
                return web.json_response(
                    {"status": "success", "data": [], "symbol": symbol, "limit": limit}
                )

            # Convert DataFrame to list of dictionaries
            data = []
            for row in df.iter_rows(named=True):
                data.append(
                    {
                        "timestamp": (
                            row["timestamp"].isoformat() if row["timestamp"] else None
                        ),
                        "open": float(row["open"]) if row["open"] is not None else None,
                        "high": float(row["high"]) if row["high"] is not None else None,
                        "low": float(row["low"]) if row["low"] is not None else None,
                        "close": (
                            float(row["close"]) if row["close"] is not None else None
                        ),
                        "volume": (
                            int(row["volume"]) if row["volume"] is not None else None
                        ),
                    }
                )

            return web.json_response(
                {
                    "status": "success",
                    "data": data,
                    "symbol": symbol,
                    "limit": limit,
                    "timeframe": timeframe_str,
                }
            )

        except Exception as e:
            self.logger.error(f"Error getting latest data for {symbol}: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def get_symbol_volatility(self, request: web.Request) -> web.Response:
        """
        Calculate volatility for a symbol.

        Query parameters:
        - days: Number of days for calculation (default: 252)
        """
        try:
            symbol = request.match_info["symbol"]
            days = int(request.query.get("days", 252))

            if not hasattr(self.data_service, "data_store"):
                return web.json_response(
                    {"status": "error", "message": "Data store not available"},
                    status=503,
                )

            # Get historical data
            from datetime import datetime, timedelta

            import numpy as np

            from shared.models import TimeFrame

            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            df = await self.data_service.data_store.load_market_data(
                ticker=symbol,
                timeframe=TimeFrame.ONE_DAY,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or len(df) < 20:
                # Return default volatility
                return web.json_response(
                    {
                        "status": "success",
                        "volatility": 0.25,
                        "symbol": symbol,
                        "days": days,
                        "note": "Default volatility used due to insufficient data",
                    }
                )

            # Calculate returns and volatility
            import polars as pl

            returns_df = df.with_columns(
                [pl.col("close").pct_change().alias("returns")]
            ).drop_nulls()

            if len(returns_df) < 10:
                return web.json_response(
                    {
                        "status": "success",
                        "volatility": 0.25,
                        "symbol": symbol,
                        "days": days,
                        "note": "Default volatility used due to insufficient returns data",
                    }
                )

            returns_std = returns_df["returns"].std()
            volatility = (
                float(returns_std) * np.sqrt(252) if returns_std is not None else 0.25
            )

            # Clamp to reasonable bounds
            volatility = max(0.05, min(2.0, volatility))

            return web.json_response(
                {
                    "status": "success",
                    "volatility": volatility,
                    "symbol": symbol,
                    "days": days,
                }
            )

        except Exception as e:
            self.logger.error(f"Error calculating volatility for {symbol}: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def get_symbol_correlation(self, request: web.Request) -> web.Response:
        """
        Calculate correlation between two symbols.

        Query parameters:
        - days: Number of days for calculation (default: 252)
        """
        try:
            symbol1 = request.match_info["symbol1"]
            symbol2 = request.match_info["symbol2"]
            days = int(request.query.get("days", 252))

            if symbol1 == symbol2:
                return web.json_response(
                    {
                        "status": "success",
                        "correlation": 1.0,
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "days": days,
                    }
                )

            if not hasattr(self.data_service, "data_store"):
                return web.json_response(
                    {"status": "error", "message": "Data store not available"},
                    status=503,
                )

            # Get historical data for both symbols
            from datetime import datetime, timedelta

            import polars as pl

            from shared.models import TimeFrame

            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            df1 = await self.data_service.data_store.load_market_data(
                ticker=symbol1,
                timeframe=TimeFrame.ONE_DAY,
                start_date=start_date,
                end_date=end_date,
            )

            df2 = await self.data_service.data_store.load_market_data(
                ticker=symbol2,
                timeframe=TimeFrame.ONE_DAY,
                start_date=start_date,
                end_date=end_date,
            )

            if df1 is None or df2 is None or len(df1) < 20 or len(df2) < 20:
                # Return default correlation
                return web.json_response(
                    {
                        "status": "success",
                        "correlation": 0.1,
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "days": days,
                        "note": "Default correlation used due to insufficient data",
                    }
                )

            # Calculate returns for both symbols
            returns1 = df1.with_columns(
                [pl.col("close").pct_change().alias("returns")]
            ).drop_nulls()

            returns2 = df2.with_columns(
                [pl.col("close").pct_change().alias("returns")]
            ).drop_nulls()

            if len(returns1) < 10 or len(returns2) < 10:
                return web.json_response(
                    {
                        "status": "success",
                        "correlation": 0.1,
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "days": days,
                        "note": "Default correlation used due to insufficient returns data",
                    }
                )

            # Merge returns data on timestamp
            merged = returns1.select(["timestamp", "returns"]).join(
                returns2.select(["timestamp", "returns"]),
                on="timestamp",
                how="inner",
                suffix="_2",
            )

            if len(merged) < 10:
                return web.json_response(
                    {
                        "status": "success",
                        "correlation": 0.1,
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "days": days,
                        "note": "Default correlation used due to insufficient overlapping data",
                    }
                )

            # Calculate correlation
            corr_matrix = merged.select(["returns", "returns_2"]).corr()
            correlation = (
                float(corr_matrix[0, 1]) if corr_matrix[0, 1] is not None else 0.1
            )

            # Clamp correlation to reasonable bounds
            correlation = max(-1.0, min(1.0, correlation))

            return web.json_response(
                {
                    "status": "success",
                    "correlation": correlation,
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "days": days,
                    "data_points": len(merged),
                }
            )

        except Exception as e:
            self.logger.error(
                f"Error calculating correlation between {symbol1} and {symbol2}: {e}"
            )
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def get_atr(self, request: web.Request) -> web.Response:
        """
        Calculate Average True Range for a symbol.

        Query parameters:
        - period: ATR period (default: 14)
        - days: Number of days of data to use (default: 30)
        """
        try:
            symbol = request.match_info["symbol"]
            period = int(request.query.get("period", 14))
            days = int(request.query.get("days", 30))

            if not hasattr(self.data_service, "data_store"):
                return web.json_response(
                    {"status": "error", "message": "Data store not available"},
                    status=503,
                )

            # Get historical OHLC data
            from datetime import datetime, timedelta

            import polars as pl

            from shared.models import TimeFrame

            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            df = await self.data_service.data_store.load_market_data(
                ticker=symbol,
                timeframe=TimeFrame.ONE_DAY,
                start_date=start_date,
                end_date=end_date,
            )

            if df is None or len(df) < period:
                # Return default ATR
                return web.json_response(
                    {
                        "status": "success",
                        "atr": 0.02,
                        "atr_percentage": 2.0,
                        "symbol": symbol,
                        "period": period,
                        "days": days,
                        "note": "Default ATR used due to insufficient data",
                    }
                )

            # Calculate True Range
            df_with_tr = df.with_columns(
                [
                    # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
                    pl.max_horizontal(
                        [
                            pl.col("high") - pl.col("low"),
                            (pl.col("high") - pl.col("close").shift(1)).abs(),
                            (pl.col("low") - pl.col("close").shift(1)).abs(),
                        ]
                    ).alias("true_range")
                ]
            ).drop_nulls()

            if len(df_with_tr) < period:
                return web.json_response(
                    {
                        "status": "success",
                        "atr": 0.02,
                        "atr_percentage": 2.0,
                        "symbol": symbol,
                        "period": period,
                        "days": days,
                        "note": "Default ATR used due to insufficient true range data",
                    }
                )

            # Calculate period-ATR
            atr_mean_val = df_with_tr["true_range"].tail(period).mean()
            atr_value = float(atr_mean_val) if atr_mean_val is not None else 0.02

            # Get current price for percentage calculation
            latest_close = df_with_tr["close"].tail(1).item()
            current_price = float(latest_close) if latest_close is not None else 1.0

            # Convert to percentage
            atr_percentage = (
                (atr_value / current_price * 100) if current_price > 0 else 2.0
            )

            # Clamp to reasonable bounds
            atr_percentage = max(0.5, min(10.0, atr_percentage))
            atr_decimal = atr_percentage / 100.0

            return web.json_response(
                {
                    "status": "success",
                    "atr": atr_decimal,
                    "atr_percentage": atr_percentage,
                    "symbol": symbol,
                    "period": period,
                    "days": days,
                    "current_price": current_price,
                    "data_points": len(df_with_tr),
                }
            )

        except Exception as e:
            self.logger.error(f"Error calculating ATR for {symbol}: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def get_real_time_quote(self, request: web.Request) -> web.Response:
        """
        Relay TwelveData real-time quote request.

        Query parameters:
        - format: Response format (default: JSON)
        """
        try:
            symbol = request.match_info["symbol"]
            _ = request.query.get("format", "JSON")

            if not hasattr(self.data_service, "twelvedata_client"):
                return web.json_response(
                    {"status": "error", "message": "TwelveData client not available"},
                    status=503,
                )

            # Get quote from TwelveData
            quote_data = await self.data_service.twelvedata_client.get_quote(symbol)

            if quote_data is None:
                return web.json_response(
                    {
                        "status": "error",
                        "message": f"No quote data available for {symbol}",
                    },
                    status=404,
                )

            return web.json_response(
                {"status": "success", "data": quote_data, "symbol": symbol}
            )

        except Exception as e:
            self.logger.error(f"Error getting real-time quote for {symbol}: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def get_time_series(self, request: web.Request) -> web.Response:
        """
        Relay TwelveData time series request.

        Query parameters:
        - interval: Time interval (5min, 15min, 1h, 1day, etc.)
        - outputsize: Number of data points (default: 30)
        - start_date: Start date (YYYY-MM-DD)
        - end_date: End date (YYYY-MM-DD)
        """
        try:
            symbol = request.match_info["symbol"]
            interval = request.query.get("interval", "5min")
            outputsize = int(request.query.get("outputsize", 30))
            start_date_str = request.query.get("start_date")
            end_date_str = request.query.get("end_date")

            if not hasattr(self.data_service, "twelvedata_client"):
                return web.json_response(
                    {"status": "error", "message": "TwelveData client not available"},
                    status=503,
                )

            # Parse dates if provided
            start_date = None
            end_date = None
            if start_date_str:
                from datetime import datetime

                start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            if end_date_str:
                from datetime import datetime

                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            # Map interval to TimeFrame
            from shared.models import TimeFrame

            interval_map = {
                "1min": TimeFrame.ONE_MINUTE,
                "5min": TimeFrame.FIVE_MINUTES,
                "15min": TimeFrame.FIFTEEN_MINUTES,
                "30min": TimeFrame.THIRTY_MINUTES,
                "1h": TimeFrame.ONE_HOUR,
                "1day": TimeFrame.ONE_DAY,
                "1week": TimeFrame.ONE_WEEK,
                "1month": TimeFrame.ONE_MONTH,
            }

            timeframe = interval_map.get(interval, TimeFrame.FIVE_MINUTES)

            # Get time series data
            market_data_list = (
                await self.data_service.twelvedata_client.get_time_series(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date,
                    outputsize=outputsize,
                )
            )

            # Convert to API response format
            data = []
            for md in market_data_list:
                data.append(
                    {
                        "datetime": md.timestamp.isoformat(),
                        "open": str(md.open),
                        "high": str(md.high),
                        "low": str(md.low),
                        "close": str(md.close),
                        "volume": str(md.volume),
                    }
                )

            return web.json_response(
                {
                    "status": "success",
                    "meta": {
                        "symbol": symbol,
                        "interval": interval,
                        "currency": "USD",
                        "exchange_timezone": "America/New_York",
                        "exchange": "NASDAQ",
                        "mic_code": "XNGS",
                        "type": "Common Stock",
                    },
                    "values": data,
                }
            )

        except Exception as e:
            self.logger.error(f"Error getting time series for {symbol}: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def get_batch_quotes(self, request: web.Request) -> web.Response:
        """
        Relay TwelveData batch quotes request.

        POST body should contain:
        {"symbols": ["AAPL", "MSFT", "GOOGL"]}
        """
        try:
            if not request.body_exists:
                return web.json_response(
                    {"status": "error", "message": "Request body required"}, status=400
                )

            data = await request.json()
            symbols = data.get("symbols", [])

            if not symbols:
                return web.json_response(
                    {"status": "error", "message": "Symbols list required"}, status=400
                )

            if not hasattr(self.data_service, "twelvedata_client"):
                return web.json_response(
                    {"status": "error", "message": "TwelveData client not available"},
                    status=503,
                )

            # Get batch real-time prices
            batch_data = (
                await self.data_service.twelvedata_client.get_batch_real_time_prices(
                    symbols
                )
            )

            # Convert to API response format
            quotes = {}
            for symbol, market_data in batch_data.items():
                if market_data:
                    quotes[symbol] = {
                        "symbol": symbol,
                        "price": str(market_data.close),
                        "timestamp": market_data.timestamp.isoformat(),
                        "open": str(market_data.open),
                        "high": str(market_data.high),
                        "low": str(market_data.low),
                        "close": str(market_data.close),
                        "volume": str(market_data.volume),
                    }
                else:
                    quotes[symbol] = {"symbol": symbol, "error": "No data available"}

            return web.json_response(
                {"status": "success", "data": quotes, "count": len(symbols)}
            )

        except Exception as e:
            self.logger.error(f"Error getting batch quotes: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def search_symbols(self, request: web.Request) -> web.Response:
        """
        Relay TwelveData symbol search request.

        Query parameters:
        - exchange: Exchange filter (optional)
        """
        try:
            query = request.match_info["query"]
            exchange = request.query.get("exchange")

            if not hasattr(self.data_service, "twelvedata_client"):
                return web.json_response(
                    {"status": "error", "message": "TwelveData client not available"},
                    status=503,
                )

            # Search instruments
            results = await self.data_service.twelvedata_client.search_instruments(
                query=query, exchange=exchange
            )

            return web.json_response(
                {"status": "success", "data": results, "count": len(results)}
            )

        except Exception as e:
            self.logger.error(f"Error searching symbols for '{query}': {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def get_technical_indicator(self, request: web.Request) -> web.Response:
        """
        Relay TwelveData technical indicator request.

        Query parameters:
        - interval: Time interval (5min, 15min, 1h, 1day, etc.)
        - time_period: Period for calculation (default: 9)
        - series_type: Price series type (default: close)
        """
        try:
            indicator = request.match_info["indicator"]
            symbol = request.match_info["symbol"]
            interval = request.query.get("interval", "1day")
            time_period = int(request.query.get("time_period", 9))
            series_type = request.query.get("series_type", "close")

            if not hasattr(self.data_service, "twelvedata_client"):
                return web.json_response(
                    {"status": "error", "message": "TwelveData client not available"},
                    status=503,
                )

            # Map interval to TimeFrame
            from shared.models import TimeFrame

            interval_map = {
                "1min": TimeFrame.ONE_MINUTE,
                "5min": TimeFrame.FIVE_MINUTES,
                "15min": TimeFrame.FIFTEEN_MINUTES,
                "30min": TimeFrame.THIRTY_MINUTES,
                "1h": TimeFrame.ONE_HOUR,
                "1day": TimeFrame.ONE_DAY,
                "1week": TimeFrame.ONE_WEEK,
                "1month": TimeFrame.ONE_MONTH,
            }

            timeframe = interval_map.get(interval, TimeFrame.ONE_DAY)

            # Get technical indicator data
            indicator_data = (
                await self.data_service.twelvedata_client.get_technical_indicators(
                    symbol=symbol,
                    indicator=indicator,
                    timeframe=timeframe,
                    time_period=time_period,
                    series_type=series_type,
                )
            )

            if indicator_data is None:
                return web.json_response(
                    {
                        "status": "error",
                        "message": f"No {indicator} data available for {symbol}",
                    },
                    status=404,
                )

            return web.json_response(
                {
                    "status": "success",
                    "data": indicator_data,
                    "meta": {
                        "symbol": symbol,
                        "indicator": indicator,
                        "interval": interval,
                        "time_period": time_period,
                        "series_type": series_type,
                    },
                }
            )

        except Exception as e:
            self.logger.error(f"Error getting {indicator} for {symbol}: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)
