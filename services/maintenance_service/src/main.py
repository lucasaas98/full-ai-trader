"""
Maintenance Service for AI Trading System

This service handles:
- Graceful shutdown procedures
- Read-only mode management
- Maintenance notifications
- Service degradation handling
- System health monitoring during maintenance
"""

import asyncio
import json
import logging
import os
import signal
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiofiles
import asyncpg
import httpx
import redis.asyncio as redis
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from monitoring.metrics import MetricsCollector
from shared.config import Config


def setup_logging(service_name: str) -> logging.Logger:
    """Set up logging configuration."""
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger = logging.getLogger(service_name)
    logger.setLevel(logging.DEBUG)

    return logger


class DatabaseManager:
    """Simple database manager for maintenance service."""

    def __init__(self, config: Config):
        self.config = config
        self.pool = None

    async def initialize(self) -> None:
        """Initialize database connection pool."""
        logger.debug(
            f"Initializing database connection pool to {self.config.database.host}:{self.config.database.port}"
        )
        self.pool = await asyncpg.create_pool(
            host=self.config.database.host,
            port=self.config.database.port,
            database=self.config.database.database,
            user=self.config.database.username,
            password=self.config.database.password,
            min_size=2,
            max_size=self.config.database.pool_size,
        )
        logger.debug("Database connection pool initialized successfully")

    async def get_connection(self) -> Any:
        """Get a database connection from the pool."""
        logger.debug("Acquiring database connection from pool")
        if not self.pool:
            logger.error("Database connection pool not initialized")
            raise RuntimeError("Database not initialized")
        return self.pool.acquire()

    async def close(self) -> None:
        """Close the database connection pool."""
        if self.pool:
            logger.debug("Closing database connection pool")
            await self.pool.close()
            logger.debug("Database connection pool closed")


# Configure logging
logger = setup_logging("maintenance_service")

# Security
security = HTTPBearer()


class MaintenanceMode(Enum):
    """Maintenance mode types"""

    NORMAL = "normal"
    READ_ONLY = "read_only"
    MAINTENANCE = "maintenance"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class ServiceStatus(Enum):
    """Service status during maintenance"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class MaintenanceState:
    """Current maintenance state"""

    mode: MaintenanceMode
    started_at: datetime
    estimated_duration: Optional[timedelta]
    reason: str
    affected_services: List[str]
    notification_sent: bool
    initiated_by: str


@dataclass
class ServiceHealthStatus:
    """Service health during maintenance"""

    service_name: str
    status: ServiceStatus
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None


class MaintenanceManager:
    """Core maintenance management functionality"""

    def __init__(
        self, db_manager: DatabaseManager, redis_client: redis.Redis, config: Config
    ):
        logger.debug("Initializing MaintenanceManager")
        self.db = db_manager
        self.redis = redis_client
        self.config = config
        self.metrics = MetricsCollector(config)

        # Current maintenance state
        self.current_state: Optional[MaintenanceState] = None

        # Service endpoints for health checks
        self.service_endpoints = {
            "data_collector": "http://data_collector:9101/health",
            "strategy_engine": "http://strategy_engine:9102/health",
            "risk_manager": "http://risk_manager:9103/health",
            "trade_executor": "http://trade_executor:9104/health",
            "scheduler": "http://scheduler:9105/health",
            "export_service": "http://export_service:9106/health",
        }

        # Notification settings
        self.gotify_url = os.getenv("GOTIFY_URL")
        self.gotify_token = os.getenv("GOTIFY_TOKEN")

        # State file for persistence
        self.state_file = "/app/data/maintenance_state.json"

    async def load_maintenance_state(self) -> None:
        """Load maintenance state from persistent storage"""
        logger.debug(f"Loading maintenance state from {self.state_file}")
        try:
            if os.path.exists(self.state_file):
                logger.debug("State file exists, loading data")
                async with aiofiles.open(self.state_file, "r") as f:
                    state_data = json.loads(await f.read())

                self.current_state = MaintenanceState(
                    mode=MaintenanceMode(state_data["mode"]),
                    started_at=datetime.fromisoformat(state_data["started_at"]),
                    estimated_duration=(
                        timedelta(seconds=state_data["estimated_duration_seconds"])
                        if state_data.get("estimated_duration_seconds")
                        else None
                    ),
                    reason=state_data["reason"],
                    affected_services=state_data["affected_services"],
                    notification_sent=state_data["notification_sent"],
                    initiated_by=state_data["initiated_by"],
                )

                logger.info(
                    f"Loaded maintenance state: {self.current_state.mode.value}"
                )
                logger.debug(
                    f"State details - reason: {self.current_state.reason}, affected_services: {self.current_state.affected_services}"
                )
            else:
                logger.debug("No existing state file found, creating default state")
                self.current_state = MaintenanceState(
                    mode=MaintenanceMode.NORMAL,
                    started_at=datetime.now(),
                    estimated_duration=None,
                    reason="System startup",
                    affected_services=[],
                    notification_sent=True,
                    initiated_by="system",
                )

        except Exception as e:
            logger.error(f"Failed to load maintenance state: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
            # Default to normal mode
            self.current_state = MaintenanceState(
                mode=MaintenanceMode.NORMAL,
                started_at=datetime.now(),
                estimated_duration=None,
                reason="State load failure - defaulting to normal",
                affected_services=[],
                notification_sent=False,
                initiated_by="system",
            )

    async def save_maintenance_state(self) -> None:
        """Save maintenance state to persistent storage"""
        logger.debug("Saving maintenance state to persistent storage")
        try:
            state_data = {
                "mode": (
                    self.current_state.mode.value if self.current_state else "normal"
                ),
                "started_at": (
                    self.current_state.started_at.isoformat()
                    if self.current_state
                    else datetime.now().isoformat()
                ),
                "estimated_duration_seconds": (
                    self.current_state.estimated_duration.total_seconds()
                    if self.current_state and self.current_state.estimated_duration
                    else None
                ),
                "reason": (
                    self.current_state.reason if self.current_state else "unknown"
                ),
                "affected_services": (
                    self.current_state.affected_services if self.current_state else []
                ),
                "notification_sent": (
                    self.current_state.notification_sent
                    if self.current_state
                    else False
                ),
                "initiated_by": (
                    self.current_state.initiated_by if self.current_state else "system"
                ),
            }

            async with aiofiles.open(self.state_file, "w") as f:
                await f.write(json.dumps(state_data, indent=2))

            logger.debug(f"Maintenance state saved to file: {self.state_file}")

            # Also store in Redis for quick access
            await self.redis.set(
                "maintenance:state",
                json.dumps(state_data),
                ex=3600,  # Expire after 1 hour
            )
            logger.debug("Maintenance state stored in Redis")

        except Exception as e:
            logger.error(f"Failed to save maintenance state: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")

    async def enter_maintenance_mode(
        self,
        mode: MaintenanceMode,
        reason: str,
        estimated_duration: Optional[timedelta] = None,
        affected_services: Optional[List[str]] = None,
        initiated_by: str = "admin",
    ) -> dict:
        """Enter specified maintenance mode"""
        logger.info(f"Entering maintenance mode: {mode.value}")
        logger.debug(
            f"Maintenance details - reason: {reason}, duration: {estimated_duration}, affected_services: {affected_services}, initiated_by: {initiated_by}"
        )

        self.current_state = MaintenanceState(
            mode=mode,
            started_at=datetime.now(),
            estimated_duration=estimated_duration,
            reason=reason,
            affected_services=affected_services or [],
            notification_sent=False,
            initiated_by=initiated_by,
        )

        await self.save_maintenance_state()

        # Send notifications
        logger.debug("Sending maintenance mode notification")
        await self.send_maintenance_notification(
            f"System entering {mode.value} mode", reason, estimated_duration
        )

        # Update service configurations
        logger.debug("Updating service maintenance status")
        await self.update_service_maintenance_status()

        # Record audit log
        await self.log_maintenance_event(
            "ENTER_MAINTENANCE",
            {
                "mode": mode.value,
                "reason": reason,
                "estimated_duration": (
                    estimated_duration.total_seconds() if estimated_duration else None
                ),
                "affected_services": affected_services,
            },
        )

    async def exit_maintenance_mode(self, initiated_by: str = "system") -> dict:
        """Exit maintenance mode and return to normal operation"""
        if self.current_state and self.current_state.mode == MaintenanceMode.NORMAL:
            logger.warning("System is already in normal mode")
            pass

        previous_mode = (
            self.current_state.mode if self.current_state else MaintenanceMode.NORMAL
        )
        logger.info(f"Exiting maintenance mode: {previous_mode.value}")
        logger.debug(f"Exit initiated by: {initiated_by}")

        self.current_state = MaintenanceState(
            mode=MaintenanceMode.NORMAL,
            started_at=datetime.now(),
            estimated_duration=None,
            reason="Maintenance completed",
            affected_services=[],
            notification_sent=False,
            initiated_by=initiated_by,
        )

        await self.save_maintenance_state()

        # Send notifications
        logger.debug("Sending maintenance exit notification")
        await self.send_maintenance_notification(
            "System maintenance completed",
            f"Exited {previous_mode.value} mode - normal operations resumed",
        )

        # Update service configurations
        logger.debug("Updating service maintenance status after exit")
        await self.update_service_maintenance_status()

        # Record audit log
        await self.log_maintenance_event(
            "EXIT_MAINTENANCE",
            {"previous_mode": previous_mode.value, "reason": "Maintenance completed"},
        )

    async def check_service_health(self) -> Dict[str, ServiceHealthStatus]:
        """Check health of all services"""
        logger.debug("Starting health check for all services")
        health_statuses = {}

        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, endpoint in self.service_endpoints.items():
                logger.debug(f"Checking health of {service_name} at {endpoint}")
                try:
                    start_time = datetime.now()
                    response = await client.get(endpoint)
                    response_time = (datetime.now() - start_time).total_seconds()

                    if response.status_code == 200:
                        status = ServiceStatus.HEALTHY
                        error_message = None
                        logger.debug(
                            f"{service_name} is healthy (response time: {response_time:.3f}s)"
                        )
                    else:
                        status = ServiceStatus.DEGRADED
                        error_message = f"HTTP {response.status_code}"
                        logger.debug(f"{service_name} is degraded: {error_message}")

                except Exception as e:
                    status = ServiceStatus.UNAVAILABLE
                    response_time = 0
                    error_message = str(e)
                    logger.debug(f"{service_name} is unavailable: {error_message}")

                health_statuses[service_name] = ServiceHealthStatus(
                    service_name=service_name,
                    status=status,
                    last_check=datetime.now(),
                    response_time=response_time,
                    error_message=error_message,
                )

        logger.debug(f"Health check completed for {len(health_statuses)} services")
        return health_statuses

    async def graceful_shutdown_service(self, service_name: str) -> bool:
        """Gracefully shutdown a specific service"""
        logger.info(f"Initiating graceful shutdown for {service_name}")
        logger.debug(
            f"Service {service_name} endpoint: {self.service_endpoints.get(service_name)}"
        )

        try:
            # Send shutdown signal to service
            endpoint = f"http://{service_name}:800{self.service_endpoints[service_name][-1]}/shutdown"
            logger.debug(f"Sending shutdown request to: {endpoint}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(endpoint, json={"graceful": True})
                    logger.debug(
                        f"Shutdown request response status: {response.status_code}"
                    )
                    if response.status_code == 200:
                        logger.info(f"Graceful shutdown initiated for {service_name}")

                        # Wait for service to stop responding
                        logger.debug(
                            f"Waiting for {service_name} to shutdown (30s timeout)"
                        )
                        for attempt in range(30):  # 30 second timeout
                            await asyncio.sleep(1)
                            try:
                                health_response = await client.get(
                                    self.service_endpoints[service_name], timeout=2.0
                                )
                                if health_response.status_code != 200:
                                    logger.debug(
                                        f"Service {service_name} stopped responding (attempt {attempt + 1})"
                                    )
                                    break
                            except Exception:
                                logger.debug(
                                    f"Service {service_name} is down (attempt {attempt + 1})"
                                )
                                break  # Service is down

                        logger.info(f"Service {service_name} shutdown completed")
                        return True

                except httpx.RequestError as e:
                    logger.warning(
                        f"Could not contact {service_name} for graceful shutdown: {e}"
                    )
                    logger.debug(f"Request error details: {type(e).__name__}: {str(e)}")
                    return False

        except Exception as e:
            logger.error(f"Failed to shutdown {service_name}: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
            return False

        return False

    async def update_service_maintenance_status(self) -> None:
        """Update all services with current maintenance status"""
        logger.debug("Updating all services with current maintenance status")
        maintenance_info = {
            "maintenance_mode": (
                self.current_state.mode.value if self.current_state else "normal"
            ),
            "started_at": (
                self.current_state.started_at.isoformat()
                if self.current_state
                else datetime.now().isoformat()
            ),
            "reason": self.current_state.reason if self.current_state else "unknown",
            "estimated_end": (
                (
                    self.current_state.started_at
                    + self.current_state.estimated_duration
                ).isoformat()
                if self.current_state and self.current_state.estimated_duration
                else None
            ),
        }

        # Store in Redis for services to check
        logger.debug("Storing maintenance info in Redis")
        await self.redis.set(
            "system:maintenance", json.dumps(maintenance_info), ex=86400  # 24 hours
        )

        # Notify each service via API if possible
        logger.debug("Notifying services of maintenance status via API")
        async with httpx.AsyncClient(timeout=10.0) as client:
            for service_name, endpoint in self.service_endpoints.items():
                try:
                    maintenance_endpoint = endpoint.replace("/health", "/maintenance")
                    logger.debug(f"Notifying {service_name} at {maintenance_endpoint}")
                    await client.put(maintenance_endpoint, json=maintenance_info)
                    logger.debug(f"Successfully notified {service_name}")
                except Exception as e:
                    logger.warning(
                        f"Could not notify {service_name} of maintenance status: {e}"
                    )
                    logger.debug(
                        f"Notification error details: {type(e).__name__}: {str(e)}"
                    )

    async def send_maintenance_notification(
        self, title: str, message: str, estimated_duration: Optional[timedelta] = None
    ) -> None:
        """Send maintenance notifications via Gotify"""
        logger.debug(f"Attempting to send notification: {title}")
        if not self.gotify_url or not self.gotify_token:
            logger.warning("Gotify not configured - skipping notification")
            pass

        try:
            notification_data = {
                "title": title,
                "message": message,
                "priority": 8,  # High priority for maintenance
                "extras": {
                    "maintenance_mode": (
                        self.current_state.mode.value
                        if self.current_state
                        else "normal"
                    ),
                    "estimated_duration": (
                        estimated_duration.total_seconds()
                        if estimated_duration
                        else None
                    ),
                    "affected_services": (
                        self.current_state.affected_services
                        if self.current_state
                        else []
                    ),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            logger.debug(f"Sending notification to Gotify at {self.gotify_url}")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.gotify_url}/message",
                    headers={"X-Gotify-Key": self.gotify_token or ""},
                    json=notification_data,
                )

                logger.debug(f"Gotify response status: {response.status_code}")
                if response.status_code == 200:
                    if self.current_state:
                        self.current_state.notification_sent = True
                        await self.save_maintenance_state()
                    logger.info("Maintenance notification sent successfully")
                else:
                    logger.error(f"Failed to send notification: {response.status_code}")
                    logger.debug(f"Gotify response content: {response.text}")

        except Exception as e:
            logger.error(f"Failed to send maintenance notification: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")

    async def log_maintenance_event(self, action: str, details: Dict[str, Any]) -> None:
        """Log maintenance events to audit trail"""
        logger.debug(f"Logging maintenance event: {action} with details: {details}")
        try:
            async with await self.db.get_connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_logs (
                        timestamp,
                        service_name,
                        action,
                        entity_type,
                        entity_id,
                        user_id,
                        changes,
                        success
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                    datetime.now(),
                    "maintenance_service",
                    action,
                    "system",
                    "maintenance",
                    self.current_state.initiated_by if self.current_state else "system",
                    json.dumps(details),
                    True,
                )
            logger.debug(f"Successfully logged maintenance event: {action}")

        except Exception as e:
            logger.error(f"Failed to log maintenance event: {e}")
            logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        logger.debug("Getting comprehensive system status")
        service_health = await self.check_service_health()

        # Calculate overall system health
        healthy_services = sum(
            1
            for status in service_health.values()
            if status.status == ServiceStatus.HEALTHY
        )
        total_services = len(service_health)

        logger.debug(
            f"Service health summary: {healthy_services}/{total_services} services healthy"
        )

        system_health = "healthy"
        if healthy_services == 0:
            system_health = "critical"
        elif healthy_services < total_services * 0.8:
            system_health = "degraded"
        elif healthy_services < total_services:
            system_health = "warning"

        logger.debug(f"Overall system health: {system_health}")

        return {
            "maintenance_mode": (
                self.current_state.mode.value if self.current_state else "normal"
            ),
            "system_health": system_health,
            "maintenance_started_at": (
                self.current_state.started_at.isoformat()
                if self.current_state
                else datetime.now().isoformat()
            ),
            "maintenance_reason": (
                self.current_state.reason if self.current_state else "normal operation"
            ),
            "estimated_end": (
                (
                    self.current_state.started_at
                    + self.current_state.estimated_duration
                ).isoformat()
                if self.current_state and self.current_state.estimated_duration
                else None
            ),
            "services": {
                name: {
                    "status": status.status.value,
                    "last_check": status.last_check.isoformat(),
                    "response_time": status.response_time,
                    "error": status.error_message,
                }
                for name, status in service_health.items()
            },
            "healthy_services": healthy_services,
            "total_services": total_services,
            "uptime_percentage": (
                (healthy_services / total_services) * 100 if total_services > 0 else 0
            ),
        }

    async def emergency_shutdown(self, reason: str, initiated_by: str = "system") -> None:
        """Perform emergency shutdown of all trading operations"""
        logger.critical(f"EMERGENCY SHUTDOWN initiated: {reason}")
        logger.debug(f"Emergency shutdown initiated by: {initiated_by}")

        await self.enter_maintenance_mode(
            MaintenanceMode.EMERGENCY_SHUTDOWN,
            f"EMERGENCY: {reason}",
            None,
            list(self.service_endpoints.keys()),
            initiated_by,
        )

        # Stop all trading operations immediately
        logger.debug("Setting emergency stop flag in Redis")
        await self.redis.set("trading:emergency_stop", "true", ex=86400)

        # Send critical notifications
        logger.debug("Sending emergency shutdown notification")
        await self.send_maintenance_notification(
            "🚨 EMERGENCY SHUTDOWN 🚨",
            f"Trading system emergency shutdown: {reason}",
            None,
        )

        # Attempt graceful shutdown of services
        logger.debug("Initiating graceful shutdown of all services")
        shutdown_tasks = []
        for service_name in self.service_endpoints.keys():
            if service_name != "maintenance_service":  # Don't shutdown ourselves
                logger.debug(f"Creating shutdown task for {service_name}")
                task = asyncio.create_task(self.graceful_shutdown_service(service_name))
                shutdown_tasks.append(task)

        # Wait for shutdowns with timeout
        logger.debug(
            f"Waiting for {len(shutdown_tasks)} services to shutdown (60s timeout)"
        )
        try:
            await asyncio.wait_for(asyncio.gather(*shutdown_tasks), timeout=60.0)
            logger.info("All services shutdown completed successfully")
        except asyncio.TimeoutError:
            logger.warning("Some services did not shutdown gracefully within timeout")


# Global service instance
maintenance_manager: Optional[MaintenanceManager] = None


async def get_maintenance_manager() -> MaintenanceManager:
    """Dependency injection for maintenance manager"""
    logger.debug("Getting maintenance manager instance")
    if maintenance_manager is None:
        logger.error("Maintenance manager not initialized")
        raise HTTPException(
            status_code=500, detail="Maintenance service not initialized"
        )
    return maintenance_manager


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API token"""
    logger.debug("Verifying API token")
    # In production, implement proper token verification
    if not credentials.credentials:
        logger.warning("Authentication failed: missing or invalid token")
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    logger.debug("Token verification successful")
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan"""
    global maintenance_manager

    try:
        # Load configuration
        logger.debug("Loading configuration")
        config = Config()

        # Initialize database
        logger.debug("Initializing database manager")
        db_manager = DatabaseManager(config)
        await db_manager.initialize()

        # Initialize Redis
        logger.debug(
            f"Initializing Redis connection to {config.redis.host}:{config.redis.port}"
        )
        redis_client = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            password=config.redis.password,
            decode_responses=True,
        )

        # Initialize maintenance manager
        logger.debug("Creating maintenance manager instance")
        maintenance_manager = MaintenanceManager(db_manager, redis_client, config)
        await maintenance_manager.load_maintenance_state()

        logger.info("Maintenance service initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to initialize maintenance service: {e}")
        logger.debug(f"Initialization error details: {type(e).__name__}: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.debug("Starting cleanup process")
        if maintenance_manager:
            logger.debug("Closing database and Redis connections")
            await maintenance_manager.db.close()
            await maintenance_manager.redis.close()
        logger.debug("Cleanup completed")


app = FastAPI(
    title="Maintenance Service",
    description="Trading system maintenance and operational control",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "maintenance_service",
    }


@app.get("/status")
async def get_status(manager: MaintenanceManager = Depends(get_maintenance_manager)) -> dict:
    """Get current system status"""
    logger.debug("System status requested")
    return await manager.get_system_status()


@app.post("/maintenance/enter")
async def enter_maintenance(
    mode: str,
    reason: str,
    estimated_minutes: Optional[int] = None,
    affected_services: Optional[str] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    manager: MaintenanceManager = Depends(get_maintenance_manager),
    token: str = Depends(verify_token),
) -> Dict[str, Any]:
    """Enter maintenance mode"""
    logger.info(f"API request to enter maintenance mode: {mode}")
    logger.debug(
        f"Request details - reason: {reason}, duration: {estimated_minutes}, services: {affected_services}"
    )

    try:
        maintenance_mode = MaintenanceMode(mode)
        estimated_duration = (
            timedelta(minutes=estimated_minutes) if estimated_minutes else None
        )
        services = affected_services.split(",") if affected_services else None

        background_tasks.add_task(
            manager.enter_maintenance_mode,
            maintenance_mode,
            reason,
            estimated_duration,
            services,
            "api_user",
        )

        logger.debug("Maintenance mode task queued successfully")
        return {
            "message": f"Entering {mode} mode",
            "estimated_duration_minutes": estimated_minutes,
            "affected_services": services,
        }

    except ValueError:
        logger.warning(f"Invalid maintenance mode provided: {mode}")
        raise HTTPException(status_code=400, detail="Invalid maintenance mode")
    except Exception as e:
        logger.error(f"Failed to enter maintenance mode: {e}")
        logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maintenance/exit")
async def exit_maintenance(
    background_tasks: BackgroundTasks = BackgroundTasks(),
    manager: MaintenanceManager = Depends(get_maintenance_manager),
    token: str = Depends(verify_token),
) -> Dict[str, Any]:
    """Exit maintenance mode"""
    logger.info("API request to exit maintenance mode")

    try:
        background_tasks.add_task(manager.exit_maintenance_mode, "api_user")
        logger.debug("Maintenance exit task queued successfully")
        return {"message": "Exiting maintenance mode"}

    except Exception as e:
        logger.error(f"Failed to exit maintenance mode: {e}")
        logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/maintenance/emergency-shutdown")
async def emergency_shutdown(
    reason: str,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    manager: MaintenanceManager = Depends(get_maintenance_manager),
    token: str = Depends(verify_token),
) -> Dict[str, Any]:
    """Perform emergency shutdown"""
    logger.critical(f"API request for emergency shutdown: {reason}")

    try:
        background_tasks.add_task(manager.emergency_shutdown, reason, "api_user")
        logger.debug("Emergency shutdown task queued successfully")
        return {"message": "Emergency shutdown initiated", "reason": reason}

    except Exception as e:
        logger.error(f"Failed to initiate emergency shutdown: {e}")
        logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/maintenance/status")
async def get_maintenance_status(
    manager: MaintenanceManager = Depends(get_maintenance_manager),
    token: str = Depends(verify_token),
) -> Dict[str, Any]:
    """Get maintenance status"""
    logger.debug("Fetching current maintenance status")

    try:
        status = await manager.get_system_status()
        logger.debug("Successfully retrieved maintenance status")

        return {
            "status": status,
            "current_mode": status.get("mode", "unknown") if status else "unknown",
            "is_maintenance_active": (
                status.get("mode") not in ["normal", None] if status else False
            ),
        }

    except Exception as e:
        logger.error(f"Failed to fetch maintenance status: {e}")
        logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/maintenance/notify")
async def send_custom_notification(
    title: str,
    message: str,
    priority: int = 5,
    manager: MaintenanceManager = Depends(get_maintenance_manager),
    token: str = Depends(verify_token),
) -> dict:
    """Send custom maintenance notification"""
    logger.info(f"Custom notification request: {title}")
    logger.debug(f"Notification details - message: {message}, priority: {priority}")

    try:
        if not manager.gotify_url or not manager.gotify_token:
            logger.warning("Gotify not configured for notifications")
            raise HTTPException(status_code=501, detail="Notifications not configured")

        notification_data = {
            "title": title,
            "message": message,
            "priority": priority,
            "extras": {
                "source": "maintenance_service",
                "timestamp": datetime.now().isoformat(),
            },
        }

        logger.debug(f"Sending notification to Gotify at {manager.gotify_url}")
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{manager.gotify_url}/message",
                headers={"X-Gotify-Key": manager.gotify_token},
                json=notification_data,
            )

            logger.debug(f"Gotify response status: {response.status_code}")
            if response.status_code == 200:
                logger.debug("Custom notification sent successfully")
                return {"message": "Notification sent successfully"}
            else:
                logger.error(
                    f"Failed to send notification: HTTP {response.status_code}"
                )
                logger.debug(f"Response content: {response.text}")
                raise HTTPException(
                    status_code=500, detail="Failed to send notification"
                )

    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Signal handlers for graceful shutdown
def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown"""
    logger.debug("Setting up signal handlers for graceful shutdown")

    def signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        logger.debug(f"Signal frame: {frame}")
        asyncio.create_task(graceful_system_shutdown())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    logger.debug("Signal handlers configured for SIGTERM and SIGINT")


async def graceful_system_shutdown() -> None:
    """Perform graceful system shutdown"""
    logger.info("Starting graceful system shutdown")
    global maintenance_manager  # noqa: F824

    if maintenance_manager:
        logger.debug("Maintenance manager available, initiating emergency shutdown")
        await maintenance_manager.emergency_shutdown(
            "System shutdown signal received", "system"
        )
    else:
        logger.warning("Maintenance manager not available during shutdown")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting maintenance service")

    # Setup signal handlers
    setup_signal_handlers()

    port = int(os.getenv("SERVICE_PORT", 9107))
    logger.debug(f"Service will run on port {port}")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)
