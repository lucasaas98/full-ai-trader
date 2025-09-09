"""
FastAPI application for the Trading Scheduler service.

This module provides REST API endpoints for controlling and monitoring
the trading system scheduler and all its components.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Response,
    WebSocket,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, generate_latest
from pydantic import BaseModel, Field

from shared.config import get_config

from .scheduler import SchedulerAPI, TaskPriority, TradingScheduler

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler_instance: Optional[TradingScheduler] = None
api_instance: Optional[SchedulerAPI] = None


class TaskTriggerRequest(BaseModel):
    """Request model for triggering tasks."""

    priority: str = Field(default="normal", description="Task priority")
    reason: str = Field(default="manual", description="Reason for triggering")


class MaintenanceModeRequest(BaseModel):
    """Request model for maintenance mode."""

    enabled: bool = Field(..., description="Enable or disable maintenance mode")


class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates."""

    config: Dict[str, Any] = Field(
        ..., description="Configuration parameters to update"
    )


class PipelineTriggerRequest(BaseModel):
    """Request model for pipeline triggering."""

    reason: str = Field(default="manual", description="Reason for triggering pipeline")


class MaintenanceTaskRequest(BaseModel):
    """Request model for maintenance task operations."""

    task_name: str = Field(..., description="Name of the maintenance task")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Optional task parameters"
    )


class MaintenanceReportRequest(BaseModel):
    """Request model for maintenance report generation."""

    report_type: str = Field(
        default="daily", description="Type of report: daily, weekly, monthly"
    )
    format_type: str = Field(
        default="json", description="Export format: json, csv, html"
    )
    include_details: bool = Field(default=True, description="Include detailed metrics")


class TradeRequest(BaseModel):
    """Request model for manual trade execution."""

    symbol: str = Field(..., description="Trading symbol")
    side: str = Field(..., description="Trade side: buy/sell")
    quantity: int = Field(..., description="Number of shares")
    order_type: str = Field(default="market", description="Order type")
    strategy: Optional[str] = Field(None, description="Strategy name")


class MaintenanceResponse(BaseModel):
    """Response model for maintenance operations."""

    task_name: str
    success: bool
    duration: float
    message: str
    details: Optional[Dict[str, Any]] = None
    files_processed: int = 0
    bytes_freed: int = 0


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Uptime in seconds")


class StatusResponse(BaseModel):
    """System status response."""

    scheduler: Dict[str, Any] = Field(..., description="Scheduler status")
    market: Dict[str, Any] = Field(..., description="Market information")
    services: Dict[str, Any] = Field(..., description="Services status")
    tasks: Dict[str, Any] = Field(..., description="Tasks status")
    queues: Dict[str, Any] = Field(..., description="Queue lengths")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    maintenance: Dict[str, Any] = Field(
        default_factory=dict, description="Maintenance system status"
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management."""
    global scheduler_instance, api_instance

    # Initialize Prometheus metrics
    init_prometheus_metrics()

    # Startup
    logger.info("Starting scheduler service...")

    try:
        config = get_config()
        scheduler_instance = TradingScheduler(config)
        api_instance = SchedulerAPI(scheduler_instance)

        await scheduler_instance.initialize()
        await scheduler_instance.start()

        logger.info("Scheduler service started successfully")

    except Exception as e:
        logger.error(f"Failed to start scheduler service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down scheduler service...")
    if scheduler_instance:
        await scheduler_instance.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Trading Scheduler API",
    description="REST API for controlling and monitoring the trading system scheduler",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
tasks_count_gauge = None
services_count_gauge = None
tasks_executed_counter = None
service_health_gauge = None


def get_scheduler() -> TradingScheduler:
    """Dependency to get scheduler instance."""
    if scheduler_instance is None:
        raise HTTPException(status_code=503, detail="Scheduler not initialized")
    return scheduler_instance


def get_api() -> SchedulerAPI:
    """Dependency to get API instance."""
    if api_instance is None:
        raise HTTPException(status_code=503, detail="API not initialized")
    return api_instance


def init_prometheus_metrics() -> None:
    """Initialize Prometheus metrics."""
    global tasks_count_gauge, services_count_gauge, tasks_executed_counter, service_health_gauge

    tasks_count_gauge = Gauge("scheduler_tasks_total", "Number of scheduled tasks")

    services_count_gauge = Gauge(
        "scheduler_services_total", "Number of registered services"
    )

    tasks_executed_counter = Counter(
        "scheduler_tasks_executed_total",
        "Total number of tasks executed",
        ["task_id", "status"],
    )

    service_health_gauge = Gauge(
        "scheduler_service_health",
        "Health status of scheduler components",
        ["component"],
    )


async def update_prometheus_metrics() -> None:
    """Update Prometheus metrics with current values."""

    try:
        if scheduler_instance:
            if tasks_count_gauge:
                tasks_count_gauge.set(len(scheduler_instance.tasks))
            if services_count_gauge:
                services_count_gauge.set(len(scheduler_instance.services))

            # Update component health
            if service_health_gauge:
                service_health_gauge.labels(component="scheduler").set(
                    1 if scheduler_instance else 0
                )
                service_health_gauge.labels(component="api").set(
                    1 if api_instance else 0
                )

    except Exception as e:
        logger.error(f"Failed to update Prometheus metrics: {e}")


@app.get("/metrics")
async def get_prometheus_metrics() -> Response:
    """Prometheus metrics endpoint."""
    try:
        # Update metrics before returning them
        await update_prometheus_metrics()

        # Generate Prometheus format
        metrics_output = generate_latest()

        return Response(content=metrics_output, media_type="text/plain; version=0.0.4")

    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return Response(
            content=f"# Error generating metrics: {str(e)}\n",
            media_type="text/plain; version=0.0.4",
            status_code=500,
        )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime=0.0,  # TODO: Track actual uptime
    )


@app.get("/status", response_model=StatusResponse)
async def get_system_status(api: SchedulerAPI = Depends(get_api)) -> StatusResponse:
    """Get comprehensive system status."""
    try:
        status = await api.get_status()
        return StatusResponse(**status)
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance")
async def get_performance_metrics(
    api: SchedulerAPI = Depends(get_api),
) -> Dict[str, Any]:
    """Get system performance metrics."""
    try:
        return await api.get_performance()
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Task management endpoints
@app.post("/tasks/{task_id}/trigger")
async def trigger_task(
    task_id: str, request: TaskTriggerRequest, api: SchedulerAPI = Depends(get_api)
) -> Dict[str, Any]:
    """Manually trigger a specific task."""
    try:
        result = await api.trigger_task(task_id, request.priority)
        return result
    except Exception as e:
        logger.error(f"Failed to trigger task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks/{task_id}/pause")
async def pause_task(
    task_id: str, api: SchedulerAPI = Depends(get_api)
) -> Dict[str, str]:
    """Pause a scheduled task."""
    try:
        result = await api.pause_task(task_id)
        return result
    except Exception as e:
        logger.error(f"Failed to pause task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks/{task_id}/resume")
async def resume_task(
    task_id: str, api: SchedulerAPI = Depends(get_api)
) -> Dict[str, str]:
    """Resume a paused task."""
    try:
        result = await api.resume_task(task_id)
        return result
    except Exception as e:
        logger.error(f"Failed to resume task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
async def list_tasks(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """List all scheduled tasks."""
    try:
        tasks_info = {}
        for task_id, task in scheduler.tasks.items():
            tasks_info[task_id] = {
                "name": task.name,
                "enabled": task.enabled,
                "priority": task.priority.value,
                "market_hours_only": task.market_hours_only,
                "last_run": task.last_run.isoformat() if task.last_run else None,
                "last_success": (
                    task.last_success.isoformat() if task.last_success else None
                ),
                "error_count": task.error_count,
                "retry_count": task.retry_count,
                "max_retries": task.max_retries,
            }
        return {"tasks": tasks_info}
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Service management endpoints
@app.post("/services/{service_name}/restart")
async def restart_service(
    service_name: str, api: SchedulerAPI = Depends(get_api)
) -> Dict[str, str]:
    """Restart a specific service."""
    try:
        result = await api.restart_service(service_name)
        return result
    except Exception as e:
        logger.error(f"Failed to restart service {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/services")
async def list_services(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """List all registered services."""
    try:
        services_info = {}
        for service_name, service in scheduler.services.items():
            services_info[service_name] = {
                "url": service.url,
                "status": service.status.value,
                "dependencies": service.dependencies,
                "error_count": service.error_count,
                "restart_count": service.restart_count,
                "last_check": (
                    service.last_check.isoformat() if service.last_check else None
                ),
            }
        return {"services": services_info}
    except Exception as e:
        logger.error(f"Failed to list services: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/services/{service_name}/start")
async def start_service(
    service_name: str, scheduler: TradingScheduler = Depends(get_scheduler)
) -> Dict[str, str]:
    """Start a specific service."""
    try:
        success = await scheduler.start_service(service_name)
        if success:
            return {"status": "success", "message": f"Service {service_name} started"}
        else:
            return {
                "status": "error",
                "message": f"Failed to start service {service_name}",
            }
    except Exception as e:
        logger.error(f"Failed to start service {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/services/{service_name}/stop")
async def stop_service(
    service_name: str, scheduler: TradingScheduler = Depends(get_scheduler)
) -> Dict[str, str]:
    """Stop a specific service."""
    try:
        success = await scheduler.stop_service(service_name)
        if success:
            return {"status": "success", "message": f"Service {service_name} stopped"}
        else:
            return {
                "status": "error",
                "message": f"Failed to stop service {service_name}",
            }
    except Exception as e:
        logger.error(f"Failed to stop service {service_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# System control endpoints
@app.post("/system/maintenance")
async def set_maintenance_mode(
    request: MaintenanceModeRequest, api: SchedulerAPI = Depends(get_api)
) -> Dict[str, str]:
    """Enable or disable maintenance mode."""
    try:
        result = await api.set_maintenance_mode(request.enabled)
        return result
    except Exception as e:
        logger.error(f"Failed to set maintenance mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system/emergency-stop")
async def emergency_stop(api: SchedulerAPI = Depends(get_api)) -> Dict[str, str]:
    """Trigger emergency stop of all trading activities."""
    try:
        result = await api.emergency_stop()
        return result
    except Exception as e:
        logger.error(f"Failed to trigger emergency stop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system/resume-trading")
async def resume_trading(api: SchedulerAPI = Depends(get_api)) -> Dict[str, str]:
    """Resume trading after emergency stop."""
    try:
        result = await api.resume_trading()
        return result
    except Exception as e:
        logger.error(f"Failed to resume trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/system/shutdown")
async def shutdown_system(
    scheduler: TradingScheduler = Depends(get_scheduler),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> Dict[str, str]:
    """Gracefully shutdown the entire system."""
    try:
        background_tasks.add_task(scheduler.shutdown)
        return {"status": "success", "message": "System shutdown initiated"}
    except Exception as e:
        logger.error(f"Failed to shutdown system: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Pipeline control endpoints
@app.post("/pipeline/trigger")
async def trigger_pipeline(
    request: PipelineTriggerRequest, api: SchedulerAPI = Depends(get_api)
) -> Dict[str, Any]:
    """Trigger the data pipeline."""
    try:
        result = await api.trigger_pipeline(request.reason)
        return result
    except Exception as e:
        logger.error(f"Failed to trigger pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pipeline/status")
async def get_pipeline_status(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get data pipeline status."""
    try:
        # Get pipeline step completion status
        pipeline_status = {}
        pipeline = getattr(scheduler, "pipeline", None)
        redis_client = getattr(scheduler, "redis", None)
        if pipeline and redis_client and hasattr(pipeline, "pipeline_steps"):
            for step in pipeline.pipeline_steps:
                key = f"pipeline:step:{step}:completed"
                completed = await redis_client.get(key)
                pipeline_status[step] = (
                    "completed" if completed is not None else "pending"
                )
        else:
            pipeline_status = {"error": "Pipeline or Redis not available"}

        return {"pipeline_status": pipeline_status}
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Configuration management endpoints
@app.get("/config")
async def get_configuration() -> Dict[str, Any]:
    """Get current configuration."""
    try:
        config = get_config()
        # Convert config to dict for JSON serialization
        config_dict = config.dict()
        return config_dict
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/reload")
async def reload_configuration(api: SchedulerAPI = Depends(get_api)) -> Dict[str, str]:
    """Hot reload configuration."""
    try:
        result = await api.update_config({})  # Trigger reload
        return result
    except Exception as e:
        logger.error(f"Failed to reload configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config/update")
async def update_configuration(
    request: ConfigUpdateRequest, api: SchedulerAPI = Depends(get_api)
) -> Dict[str, str]:
    """Update configuration parameters."""
    try:
        result = await api.update_config(request.config)
        return result
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Market information endpoints
@app.get("/market/hours")
async def get_market_hours(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get market hours information."""
    try:
        market_hours = scheduler.market_hours
        now = datetime.now(market_hours.timezone)

        current_session = await market_hours.get_current_session()
        next_open = await market_hours.get_next_market_open()
        next_close = await market_hours.get_next_market_close()

        return {
            "current_session": current_session.value,
            "is_trading_day": market_hours.is_trading_day(now.date()),
            "is_market_open": await market_hours.is_market_open(),
            "next_market_open": next_open.isoformat() if next_open else None,
            "next_market_close": next_close.isoformat() if next_close else None,
            "time_until_open": str(
                getattr(market_hours, "time_until_market_open", lambda: None)()
            ),
            "time_until_close": str(
                getattr(market_hours, "time_until_market_close", lambda: None)()
            ),
            "timezone": str(market_hours.timezone),
        }
    except Exception as e:
        logger.error(f"Failed to get market hours: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/calendar")
async def get_market_calendar(
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get market calendar for date range."""
    try:
        from datetime import datetime

        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        calendar_data = []
        current_date = start
        while current_date <= end:
            calendar_data.append(
                {
                    "date": current_date.isoformat(),
                    "is_trading_day": scheduler.market_hours.is_trading_day(
                        current_date
                    ),
                    "day_of_week": current_date.strftime("%A"),
                }
            )
            current_date += timedelta(days=1)

        return {"calendar": calendar_data}
    except Exception as e:
        logger.error(f"Failed to get market calendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring endpoints
@app.get("/metrics")
async def get_system_metrics(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get detailed system metrics."""
    try:
        monitor = getattr(scheduler, "monitor", None)
        if not monitor:
            raise HTTPException(status_code=503, detail="Monitor not available")
        metrics = monitor.get_system_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/historical")
async def get_historical_metrics(
    hours: int = Query(default=24, description="Hours of historical data"),
    scheduler: TradingScheduler = Depends(get_scheduler),
):
    """Get historical system metrics."""
    try:
        # Get historical metrics from Redis time series
        now = datetime.now().timestamp()
        start_time = now - (hours * 3600)

        redis_client = getattr(scheduler, "redis", None)
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis client not available")

        metrics = await redis_client.zrangebyscore(
            "system:metrics:timeseries", start_time, now, withscores=True
        )

        historical_data = []
        for metric_data, timestamp in metrics:
            try:
                # Parse metric data (stored as string)
                import ast

                parsed_data = ast.literal_eval(
                    metric_data.decode()
                    if isinstance(metric_data, bytes)
                    else metric_data
                )
                parsed_data["timestamp"] = timestamp
                historical_data.append(parsed_data)
            except Exception:
                continue

        return {"metrics": historical_data}
    except Exception as e:
        logger.error(f"Failed to get historical metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/logs")
async def get_logs(
    lines: int = Query(100, description="Number of log lines to retrieve"),
    level: str = Query("INFO", description="Minimum log level"),
) -> Dict[str, Any]:
    """Get recent system logs."""
    try:
        # This would typically read from a centralized logging system
        # For now, return a placeholder response
        return {
            "logs": [
                f"{datetime.now().isoformat()} - INFO - Scheduler running normally",
                f"{datetime.now().isoformat()} - INFO - All services healthy",
            ]
        }
    except Exception as e:
        logger.error(f"Failed to get logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/alerts")
async def get_alerts(
    severity: str = Query(default=None, description="Filter alerts by severity"),
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get system alerts."""
    try:
        # Get alerts from Redis
        alerts_key = "system:alerts"
        if severity:
            alerts_key = f"system:alerts:{severity}"

        redis_client = getattr(scheduler, "redis", None)
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis client not available")

        alerts_data = await redis_client.lrange(alerts_key, 0, -1)
        alerts = []

        for alert_data in alerts_data:
            try:
                import ast

                alert = ast.literal_eval(
                    alert_data.decode() if isinstance(alert_data, bytes) else alert_data
                )
                alerts.append(alert)
            except Exception:
                continue

        return {"alerts": alerts}
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/alerts")
async def clear_alerts(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, str]:
    """Clear all system alerts."""
    try:
        redis_client = getattr(scheduler, "redis", None)
        if not redis_client:
            raise HTTPException(status_code=503, detail="Redis client not available")

        # Clear all alert queues
        for severity in ["low", "medium", "high", "critical"]:
            await redis_client.delete(f"system:alerts:{severity}")
        await redis_client.delete("system:alerts")

        return {"status": "success", "message": "All alerts cleared"}
    except Exception as e:
        logger.error(f"Failed to clear alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Maintenance endpoints
@app.post("/maintenance/tasks/{task_name}/run")
async def run_maintenance_task(
    task_name: str,
    request: MaintenanceTaskRequest,
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, str]:
    """Run a specific maintenance task."""
    try:
        if hasattr(scheduler, "maintenance_manager"):
            maintenance_manager = getattr(scheduler, "maintenance_manager", None)
            if not maintenance_manager:
                raise HTTPException(
                    status_code=503, detail="Maintenance manager not available"
                )
            result = await maintenance_manager.resume_scheduled_task(task_name)
            return MaintenanceResponse(
                task_name=result.task_name,
                success=result.success,
                duration=result.duration,
                message=result.message,
                details=result.details,
                files_processed=result.files_processed,
                bytes_freed=result.bytes_freed,
            )
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance manager not available"
            )
    except Exception as e:
        logger.error(f"Failed to run maintenance task {task_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maintenance/run-all")
async def run_all_maintenance_tasks(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Run all maintenance tasks."""
    try:
        if hasattr(scheduler, "maintenance_manager"):
            maintenance_manager = getattr(scheduler, "maintenance_manager", None)
            if not maintenance_manager:
                raise HTTPException(
                    status_code=503, detail="Maintenance manager not available"
                )
            results = await maintenance_manager.run_all_tasks()

            # Convert results to response format
            response_results = {}
            for task_name, result in results.items():
                response_results[task_name] = {
                    "success": result.success,
                    "duration": result.duration,
                    "message": result.message,
                    "bytes_freed": result.bytes_freed,
                }

            return {"results": response_results}
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance manager not available"
            )
    except Exception as e:
        logger.error(f"Failed to run all maintenance tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/maintenance/status")
async def get_maintenance_status(scheduler: TradingScheduler = Depends(get_scheduler)):
    """Get maintenance system status."""
    try:
        if hasattr(scheduler, "maintenance_manager"):
            manager = getattr(scheduler, "maintenance_manager", None)
            if not manager:
                raise HTTPException(
                    status_code=503, detail="Maintenance manager not available"
                )

            stats: Dict[str, Any] = {
                "is_running": getattr(manager, "is_running", False),
                "current_task": getattr(manager, "current_task", None),
                "maintenance_tasks": list(
                    getattr(manager, "maintenance_tasks", {}).keys()
                ),
                "last_run": getattr(manager, "last_run", None),
            }

            # Get maintenance history
            history = manager.get_maintenance_history()
            if history:
                stats["last_maintenance_cycle"] = history[0]

            return stats
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance manager not available"
            )
    except Exception as e:
        logger.error(f"Failed to get maintenance status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/maintenance/history")
async def get_maintenance_history(
    days: int = Query(default=30, description="Number of days to retrieve"),
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get maintenance task history."""
    try:
        if hasattr(scheduler, "maintenance_manager"):
            maintenance_manager = getattr(scheduler, "maintenance_manager", None)
            if not maintenance_manager:
                raise HTTPException(
                    status_code=503, detail="Maintenance manager not available"
                )
            # Get maintenance metrics if available but don't fail if not
            try:
                if hasattr(scheduler, "monitor"):
                    monitor = getattr(scheduler, "monitor", None)
                    if monitor:
                        pass  # Metrics available but not needed for history
            except Exception:
                pass  # Ignore metrics errors for history endpoint
            history = maintenance_manager.get_maintenance_history()
            return {"history": history}
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance manager not available"
            )
    except Exception as e:
        logger.error(f"Failed to get maintenance history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maintenance/reports/generate")
async def generate_maintenance_report(
    request: MaintenanceReportRequest,
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Generate maintenance report."""
    try:
        if hasattr(scheduler, "maintenance_manager"):
            from .maintenance import MaintenanceReportGenerator

            maintenance_manager = getattr(scheduler, "maintenance_manager", None)
            if not maintenance_manager:
                raise HTTPException(
                    status_code=503, detail="Maintenance manager not available"
                )
            report_generator = MaintenanceReportGenerator(maintenance_manager)

            if request.report_type == "daily":
                report = await report_generator.generate_daily_report()
            elif request.report_type == "weekly":
                report = await report_generator.generate_weekly_report()
            else:
                raise HTTPException(status_code=400, detail="Invalid report type")

            # Export if requested
            if request.format_type != "json":
                export_path = await report_generator.export_report_to_file(
                    report, request.format_type
                )
                report["export_path"] = export_path

            return {"report": report}
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance manager not available"
            )
    except Exception as e:
        logger.error(f"Failed to generate maintenance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/maintenance/schedule")
async def get_maintenance_schedule(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get maintenance task schedule."""
    try:
        if hasattr(scheduler, "maintenance_scheduler"):
            maintenance_manager = getattr(scheduler, "maintenance_manager", None)
            if not maintenance_manager:
                raise HTTPException(
                    status_code=503, detail="Maintenance manager not available"
                )
            schedule = maintenance_manager.get_maintenance_schedule()
            next_tasks = maintenance_manager.get_next_scheduled_tasks()

            return {"scheduled_tasks": schedule, "next_24_hours": next_tasks}
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance scheduler not available"
            )
    except Exception as e:
        logger.error(f"Failed to get maintenance schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maintenance/schedule/{schedule_id}/pause")
async def pause_maintenance_task(
    schedule_id: str, scheduler: TradingScheduler = Depends(get_scheduler)
) -> Dict[str, str]:
    """Pause a scheduled maintenance task."""
    try:
        if hasattr(scheduler, "maintenance_manager"):
            maintenance_manager = getattr(scheduler, "maintenance_manager", None)
            if not maintenance_manager:
                raise HTTPException(
                    status_code=503, detail="Maintenance manager not available"
                )
            await maintenance_manager.pause_scheduled_task(schedule_id)
            return {
                "status": "success",
                "message": f"Maintenance task {schedule_id} paused",
            }
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance scheduler not available"
            )
    except Exception as e:
        logger.error(f"Failed to pause maintenance task {schedule_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maintenance/schedule/{schedule_id}/resume")
async def resume_maintenance_task(
    schedule_id: str, scheduler: TradingScheduler = Depends(get_scheduler)
) -> Dict[str, str]:
    """Resume a scheduled maintenance task."""
    try:
        if hasattr(scheduler, "maintenance_manager"):
            maintenance_manager = getattr(scheduler, "maintenance_manager", None)
            if not maintenance_manager:
                raise HTTPException(
                    status_code=503, detail="Maintenance manager not available"
                )
            await maintenance_manager.resume_scheduled_task(schedule_id)
            return {
                "status": "success",
                "message": f"Maintenance task {schedule_id} resumed",
            }
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance scheduler not available"
            )
    except Exception as e:
        logger.error(f"Failed to resume maintenance task {schedule_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maintenance/smart-run")
async def run_smart_maintenance(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Run intelligent maintenance based on system analysis."""
    try:
        if hasattr(scheduler, "maintenance_scheduler"):
            maintenance_manager = getattr(scheduler, "maintenance_manager", None)
            if not maintenance_manager:
                raise HTTPException(
                    status_code=503, detail="Maintenance manager not available"
                )
            results = await maintenance_manager.run_smart_maintenance()
            return {"results": results}
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance scheduler not available"
            )
    except Exception as e:
        logger.error(f"Failed to run smart maintenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/maintenance/metrics")
async def get_maintenance_metrics(
    task_name: Optional[str] = Query(None, description="Specific task name"),
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get maintenance task metrics."""
    try:
        maintenance_manager = getattr(scheduler, "maintenance_manager", None)
        if maintenance_manager:
            monitor = getattr(maintenance_manager, "monitor", None)
            if monitor:
                metrics = await monitor.get_maintenance_metrics(task_name)
                return {"metrics": metrics}
            else:
                return {"metrics": {"error": "Maintenance monitor not available"}}
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance manager not available"
            )
    except Exception as e:
        logger.error(f"Failed to get maintenance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/maintenance/statistics")
async def get_maintenance_statistics(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get comprehensive maintenance statistics."""
    try:
        maintenance_manager = getattr(scheduler, "maintenance_manager", None)
        if maintenance_manager:
            stats: Dict[str, Any] = getattr(
                maintenance_manager, "get_maintenance_statistics", lambda: {}
            )()
            return {"statistics": stats}
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance manager not available"
            )
    except Exception as e:
        logger.error(f"Failed to get maintenance statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/maintenance/dashboard")
async def get_maintenance_dashboard(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get maintenance dashboard data."""
    try:
        if hasattr(scheduler, "maintenance_manager") and hasattr(
            scheduler, "maintenance_scheduler"
        ):
            dashboard_data: Dict[str, Any] = {
                "system_health": {},
                "recent_tasks": [],
                "scheduled_tasks": {},
                "performance_metrics": {},
                "alerts": [],
                "recommendations": [],
            }

            # Get recent maintenance history
            maintenance_manager = getattr(scheduler, "maintenance_manager", None)
            if not maintenance_manager:
                raise HTTPException(
                    status_code=503, detail="Maintenance manager not available"
                )
            history = maintenance_manager.get_maintenance_history()
            dashboard_data["recent_tasks"] = history

            # Get scheduled tasks
            if hasattr(maintenance_manager, "get_maintenance_schedule"):
                dashboard_data["scheduled_tasks"] = (
                    maintenance_manager.get_maintenance_schedule()
                )
            else:
                dashboard_data["scheduled_tasks"] = []

            # Get maintenance metrics
            monitor = getattr(maintenance_manager, "monitor", None)
            if monitor and hasattr(monitor, "get_maintenance_metrics"):
                metrics = await monitor.get_maintenance_metrics()
                dashboard_data["performance_metrics"] = metrics
            else:
                dashboard_data["performance_metrics"] = {
                    "error": "Monitor not available"
                }

            # Get system health
            try:
                import psutil

                dashboard_data["system_health"] = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage("/").percent,
                    "load_average": (
                        psutil.getloadavg()
                        if hasattr(psutil, "getloadavg")
                        else [0, 0, 0]
                    ),
                }
            except Exception:
                dashboard_data["system_health"] = {
                    "error": "Unable to collect system metrics"
                }

            # Get recent alerts
            redis_client = getattr(scheduler, "redis", None)
            if not redis_client:
                raise HTTPException(
                    status_code=503, detail="Redis client not available"
                )
            alerts_data = await redis_client.lrange("system:alerts", 0, -1)
            alerts = []
            for item in alerts_data:
                try:
                    alert = json.loads(
                        item.decode() if isinstance(item, bytes) else item
                    )
                    alerts.append(alert)
                except Exception:
                    continue
            dashboard_data["alerts"] = alerts

            return {"dashboard": dashboard_data}
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance system not available"
            )
    except Exception as e:
        logger.error(f"Failed to get maintenance dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maintenance/emergency")
async def run_emergency_maintenance(
    task_name: Optional[str] = Query(None, description="Specific emergency task"),
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Run emergency maintenance tasks."""
    try:
        if hasattr(scheduler, "maintenance_manager"):
            if task_name:
                maintenance_manager = getattr(scheduler, "maintenance_manager", None)
                if not maintenance_manager:
                    raise HTTPException(
                        status_code=503, detail="Maintenance manager not available"
                    )
                run_task_func = getattr(maintenance_manager, "run_task", None)
                if run_task_func:
                    result = await run_task_func(task_name)
                else:

                    class Result:
                        def __init__(self):
                            self.success = False
                            self.message = "Task runner not available"
                            self.duration = 0

                    result = Result()
                return {
                    "emergency_task": task_name,
                    "success": result.success,
                    "message": result.message,
                    "duration": result.duration,
                }
            else:
                # Run all emergency tasks
                emergency_tasks = [
                    "system_health_check",
                    "cache_cleanup",
                    "portfolio_reconciliation",
                ]
                results = {}
                maintenance_manager = getattr(scheduler, "maintenance_manager", None)
                if maintenance_manager:
                    run_all_func = getattr(maintenance_manager, "run_all_tasks", None)
                    if run_all_func:
                        all_results = await run_all_func()
                        results = all_results if isinstance(all_results, dict) else {}
                    else:
                        # Run tasks individually if run_all_tasks not available
                        for task in emergency_tasks:
                            run_task_func = getattr(
                                maintenance_manager, "run_task", None
                            )
                            if run_task_func:
                                try:
                                    result = await run_task_func(task)
                                    results[task] = {
                                        "success": getattr(result, "success", False),
                                        "message": getattr(
                                            result, "message", "Task completed"
                                        ),
                                        "duration": getattr(result, "duration", 0),
                                    }
                                except Exception as e:
                                    results[task] = {
                                        "success": False,
                                        "message": str(e),
                                        "duration": 0,
                                    }
                            else:
                                results[task] = {
                                    "success": False,
                                    "message": "Task runner not available",
                                    "duration": 0,
                                }
                else:
                    for task in emergency_tasks:
                        results[task] = {
                            "success": False,
                            "message": "Maintenance manager not available",
                            "duration": 0,
                        }
                return {"emergency_results": results}
        else:
            raise HTTPException(
                status_code=503, detail="Maintenance manager not available"
            )
    except Exception as e:
        logger.error(f"Failed to run emergency maintenance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Queue management endpoints
@app.get("/queues")
async def get_queue_status(
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, Any]:
    """Get task queue status."""
    try:
        task_queue = getattr(scheduler, "task_queue", None)
        if not task_queue:
            raise HTTPException(status_code=503, detail="Task queue not available")
        queue_lengths = await task_queue.get_queue_lengths()
        return {"queues": queue_lengths}
    except Exception as e:
        logger.error(f"Failed to get queue status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/queues/clear")
async def clear_queues(
    priority: str = Query(default="all", description="Priority queue to clear"),
    scheduler: TradingScheduler = Depends(get_scheduler),
) -> Dict[str, str]:
    """Clear task queues."""
    try:
        task_queue = getattr(scheduler, "task_queue", None)
        redis_client = getattr(scheduler, "redis", None)
        if not task_queue or not redis_client:
            raise HTTPException(
                status_code=503, detail="Task queue or Redis not available"
            )

        if priority == "all":
            queues = getattr(task_queue, "queues", {})
            for queue_priority, queue_name in queues.items():
                await redis_client.delete(queue_name)
        else:
            try:
                queue_priority = TaskPriority(priority)
                queues = getattr(task_queue, "queues", {})
                queue_name = queues[queue_priority]
                await redis_client.delete(queue_name)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid priority level")

        return {"status": "success", "message": f"Cleared {priority} queues"}
    except Exception as e:
        logger.error(f"Failed to clear queues: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Trade management endpoints (proxy to trade executor)
@app.get("/positions")
async def get_positions() -> Dict[str, Any]:
    """Get current positions."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://trade-executor:9104/positions")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/portfolio")
async def get_portfolio() -> Dict[str, Any]:
    """Get portfolio summary."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://trade-executor:9104/portfolio")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/trades/manual")
async def execute_manual_trade(request: TradeRequest) -> Dict[str, Any]:
    """Execute a manual trade."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://trade-executor:9104/trades/manual", json=request.dict()
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to execute manual trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trades/recent")
async def get_recent_trades(
    limit: int = Query(default=50, description="Number of trades to retrieve")
) -> Dict[str, Any]:
    """Get recent trades."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://trade-executor:9104/trades/recent?limit={limit}"
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get recent trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trades/export")
async def export_trades(
    format: str = Query(default="csv", description="Export format"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)"),
) -> Dict[str, Any]:
    """Export trades for external analysis."""
    try:
        params = {"format": format}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://trade-executor:9104/trades/export", params=params
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to export trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Strategy management endpoints (proxy to strategy engine)
@app.get("/strategies")
async def get_strategies() -> Dict[str, Any]:
    """Get all strategies and their status."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://strategy-engine:9102/strategies")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategies/{strategy_name}/toggle")
async def toggle_strategy(strategy_name: str, enabled: bool) -> Dict[str, str]:
    """Enable or disable a specific strategy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://strategy-engine:9102/strategies/{strategy_name}/toggle",
                json={"enabled": enabled},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to toggle strategy {strategy_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/strategies/{strategy_name}/backtest")
async def run_strategy_backtest(
    strategy_name: str,
    start_date: str = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(default=None, description="End date (YYYY-MM-DD)"),
    symbols: str = Query(default=None, description="Comma-separated symbols"),
):
    """Run backtest for a specific strategy."""
    try:
        params = {"strategy": strategy_name}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if symbols:
            params["symbols"] = ",".join(symbols.split(","))

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://strategy-engine:9102/backtest", json=params, timeout=300.0
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error("Failed to run backtest: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Risk management endpoints (proxy to risk manager)
@app.get("/risk/status")
async def get_risk_status() -> Dict[str, Any]:
    """Get current risk status."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://risk-manager:9103/risk/status")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get risk status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/limits")
async def get_risk_limits() -> Dict[str, Any]:
    """Get current risk limits."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://risk-manager:9103/risk/limits")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get risk limits: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk/alerts")
async def get_risk_alerts() -> Dict[str, Any]:
    """Get current risk alerts."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://risk-manager:9103/risk/alerts")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to get risk alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Data collection endpoints (proxy to data collector)
@app.post("/data/update")
async def trigger_data_update(
    symbols: str = Query(default=None, description="Comma-separated symbols"),
    timeframe: str = Query(default="1m", description="Data timeframe"),
) -> Dict[str, str]:
    """Trigger data update for specific symbols."""
    try:
        params = {"timeframe": timeframe}
        if symbols:
            params["symbols"] = ",".join(symbols.split(","))

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://data-collector:9101/market-data/update", json=params
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to trigger data update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/screener/scan")
async def trigger_screener_scan() -> Dict[str, str]:
    """Trigger FinViz screener scan."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post("http://data-collector:9101/finviz/scan")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to trigger screener scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time updates


@app.websocket("/ws/status")
async def websocket_status_updates(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time status updates."""
    await websocket.accept()

    try:
        while True:
            if scheduler_instance:
                status = await scheduler_instance.get_system_status()
                await websocket.send_json(status)

            await asyncio.sleep(5)  # Send updates every 5 seconds

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Any, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Any, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
        },
    )


# Startup event
@app.on_event("startup")
async def startup_event() -> None:
    """Application startup event."""
    logger.info("Scheduler API starting up...")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Application shutdown event."""
    logger.info("Scheduler API shutting down...")


# Main application entry point
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


if __name__ == "__main__":
    config = get_config()
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=config.is_development,
        log_level=config.logging.level.lower(),
    )
