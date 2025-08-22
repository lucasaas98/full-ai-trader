"""
HTTP server for the data collection service.

This module provides a lightweight HTTP server that exposes health check
endpoints and basic service information for integration with the scheduler
and other services in the trading system.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from aiohttp import web, web_response
from aiohttp.web_request import Request
from aiohttp.web_response import Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest


logger = logging.getLogger(__name__)


class DataCollectorHTTPServer:
    """HTTP server for data collector service endpoints."""

    def __init__(self, data_service=None, port: int = 8001, host: str = "0.0.0.0"):
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

        # Add middleware for CORS and error handling
        app.middlewares.append(self.cors_handler)
        app.middlewares.append(self.error_handler)

        return app

    async def start(self):
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

    async def stop(self):
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
    async def cors_handler(self, request: Request, handler):
        """CORS middleware."""
        response = await handler(request)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    @web.middleware
    async def error_handler(self, request: Request, handler):
        """Error handling middleware."""
        try:
            return await handler(request)
        except Exception as e:
            self.logger.error(f"HTTP request error: {e}")
            return web.json_response(
                {
                    "error": "Internal server error",
                    "message": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                status=500
            )

    async def index(self, request: Request) -> Response:
        """Root endpoint with basic service information."""
        return web.json_response({
            "service": "data_collector",
            "version": "1.0.0",
            "status": "running",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoints": {
                "health": "/health",
                "status": "/status",
                "metrics": "/metrics",
                "metrics_json": "/metrics/json",
                "info": "/info"
            }
        })

    async def health_check(self, request: Request) -> Response:
        """Health check endpoint for service monitoring."""
        try:
            if not self.data_service:
                return web.json_response(
                    {
                        "status": "unhealthy",
                        "message": "Data service not available",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    status=503
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
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                status=500
            )

    async def status(self, request: Request) -> Response:
        """Detailed status endpoint."""
        try:
            if not self.data_service:
                return web.json_response(
                    {"error": "Data service not available"},
                    status=503
                )

            # Get service status
            status = await self.data_service.get_service_status()

            return web.json_response({
                "service": "data_collector",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": status
            })

        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return web.json_response(
                {"error": f"Status check failed: {str(e)}"},
                status=500
            )

    async def prometheus_metrics(self, request: Request) -> Response:
        """Prometheus-style metrics endpoint."""
        try:
            if not self.data_service:
                return web.Response(
                    text="# Data service not available\n",
                    content_type="text/plain; version=0.0.4",
                    status=503
                )

            # Update Prometheus metrics with current values
            await self._update_prometheus_metrics()

            # Generate Prometheus format
            metrics_output = generate_latest()

            return web.Response(
                text=metrics_output.decode('utf-8'),
                content_type="text/plain; version=0.0.4"
            )

        except Exception as e:
            self.logger.error(f"Prometheus metrics collection failed: {e}")
            return web.Response(
                text=f"# Error collecting metrics: {str(e)}\n",
                content_type="text/plain; version=0.0.4",
                status=500
            )

    async def metrics(self, request: Request) -> Response:
        """JSON metrics endpoint for monitoring systems."""
        try:
            if not self.data_service:
                return web.json_response(
                    {"error": "Data service not available"},
                    status=503
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
                    "statistics": status.get("statistics", {})
                }
            }

            return web.json_response(metrics)

        except Exception as e:
            self.logger.error(f"Metrics collection failed: {e}")
            return web.json_response(
                {"error": f"Metrics collection failed: {str(e)}"},
                status=500
            )

    async def info(self, request: Request) -> Response:
        """Service information endpoint."""
        try:
            if not self.data_service:
                config_info = {"error": "Data service not available"}
            else:
                # Get configuration information (sanitized)
                config = self.data_service.config.dict() if hasattr(self.data_service, 'config') else {}

                # Remove sensitive information
                config_info = {
                    k: v for k, v in config.items()
                    if k not in ['api_key', 'password', 'secret']
                }

            return web.json_response({
                "service": "data_collector",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "configuration": config_info,
                "server_info": {
                    "host": self.host,
                    "port": self.port,
                    "started": datetime.now(timezone.utc).isoformat()
                }
            })

        except Exception as e:
            self.logger.error(f"Info request failed: {e}")
            return web.json_response(
                {"error": f"Info request failed: {str(e)}"},
                status=500
            )

    async def _get_health_info(self) -> Dict[str, Any]:
        """Get comprehensive health information from the data service."""
        health_info = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": "data_collector",
            "status": "unknown",
            "components": {},
            "errors": []
        }

        try:
            # Get service status
            service_status = await self.data_service.get_service_status()

            # Check if service is running
            is_running = service_status.get("is_running", False)
            health_info["components"]["service"] = {
                "status": "healthy" if is_running else "unhealthy",
                "running": is_running,
                "active_tickers": service_status.get("active_tickers_count", 0)
            }

            # Check individual components if available
            if hasattr(self.data_service, 'finviz_screener') and self.data_service.finviz_screener:
                try:
                    finviz_healthy = await self.data_service.finviz_screener.validate_connection()
                    health_info["components"]["finviz"] = {
                        "status": "healthy" if finviz_healthy else "degraded"
                    }
                    if not finviz_healthy:
                        health_info["errors"].append("FinViz connection issues")
                except Exception:
                    health_info["components"]["finviz"] = {"status": "unknown"}

            if hasattr(self.data_service, 'twelvedata_client') and self.data_service.twelvedata_client:
                try:
                    twelvedata_healthy = await self.data_service.twelvedata_client.test_connection()
                    health_info["components"]["twelvedata"] = {
                        "status": "healthy" if twelvedata_healthy else "degraded"
                    }
                    if not twelvedata_healthy:
                        health_info["errors"].append("TwelveData connection issues")
                except Exception:
                    health_info["components"]["twelvedata"] = {"status": "unknown"}

            if hasattr(self.data_service, 'redis_client') and self.data_service.redis_client:
                try:
                    redis_health = await self.data_service.redis_client.health_check()
                    health_info["components"]["redis"] = {
                        "status": "healthy" if redis_health.get("connected", False) else "unhealthy",
                        "connected": redis_health.get("connected", False)
                    }
                    if not redis_health.get("connected", False):
                        health_info["errors"].append("Redis connection failed")
                except Exception:
                    health_info["components"]["redis"] = {"status": "unknown"}
                    health_info["errors"].append("Redis health check failed")

            # Determine overall health status
            component_statuses = [
                comp.get("status") for comp in health_info["components"].values()
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
            health_info["errors"].append(f"Health check failed: {str(e)}")

        return health_info

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        # Service status metrics
        self.active_tickers_gauge = Gauge(
            'data_collector_active_tickers_total',
            'Number of active tickers being tracked'
        )

        self.scheduled_jobs_gauge = Gauge(
            'data_collector_scheduled_jobs_total',
            'Number of scheduled jobs'
        )

        self.service_running_gauge = Gauge(
            'data_collector_service_running',
            'Whether the data collection service is running (1=running, 0=stopped)'
        )

        # Statistics metrics
        self.screener_runs_counter = Counter(
            'data_collector_screener_runs_total',
            'Total number of screener runs'
        )

        self.data_updates_counter = Counter(
            'data_collector_data_updates_total',
            'Total number of data updates'
        )

        self.records_saved_counter = Counter(
            'data_collector_records_saved_total',
            'Total number of records saved'
        )

        self.errors_counter = Counter(
            'data_collector_errors_total',
            'Total number of errors encountered'
        )

        # Component health metrics
        self.component_health_gauge = Gauge(
            'data_collector_component_health',
            'Health status of components (1=healthy, 0=unhealthy)',
            ['component']
        )

        # HTTP request metrics
        self.http_requests_counter = Counter(
            'data_collector_http_requests_total',
            'Total HTTP requests',
            ['endpoint', 'method', 'status']
        )

        self.http_request_duration = Histogram(
            'data_collector_http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['endpoint', 'method']
        )

    async def _update_prometheus_metrics(self):
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
                    self.component_health_gauge.labels(component=component_name).set(health_value)

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
            if hasattr(self.service, '_update_price_data'):
                await self.service._update_price_data(timeframe)

                return web.json_response({
                    "status": "success",
                    "message": f"Market data update triggered for {timeframe_str}",
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                return web.json_response(
                    {"status": "error", "message": "Data collection service not available"},
                    status=503
                )

        except Exception as e:
            self.logger.error(f"Error triggering market data update: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500
            )

    async def trigger_finviz_scan(self, request: web.Request) -> web.Response:
        """
        Trigger FinViz screener scan.
        """
        try:
            # Trigger the scan
            if hasattr(self.service, '_run_finviz_scan'):
                await self.service._run_finviz_scan()

                return web.json_response({
                    "status": "success",
                    "message": "FinViz scan triggered",
                    "timestamp": datetime.utcnow().isoformat()
                })
            else:
                return web.json_response(
                    {"status": "error", "message": "FinViz screener not available"},
                    status=503
                )

        except Exception as e:
            self.logger.error(f"Error triggering FinViz scan: {e}")
            return web.json_response(
                {"status": "error", "message": str(e)},
                status=500
            )
