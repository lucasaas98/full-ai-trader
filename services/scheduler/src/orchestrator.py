"""
Service Orchestrator for the Trading System.

This module manages the lifecycle, dependencies, and health of all trading system services.
It provides service discovery, dependency resolution, graceful startup/shutdown sequences,
and automatic recovery mechanisms.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
from circuitbreaker import circuit
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


class HealthCheckStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    TIMEOUT = "timeout"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ServiceDependency:
    """Service dependency information."""
    service_name: str
    required: bool = True
    startup_delay: float = 0.0  # Seconds to wait after dependency starts


@dataclass
class HealthCheck:
    """Health check configuration."""
    endpoint: str = "/health"
    timeout: float = 5.0
    interval: float = 30.0
    failure_threshold: int = 3
    success_threshold: int = 2
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check: Optional[datetime] = None
    status: HealthCheckStatus = HealthCheckStatus.UNKNOWN


@dataclass
class ServiceConfiguration:
    """Service configuration and metadata."""
    name: str
    url: str
    port: int
    health_check: HealthCheck = field(default_factory=HealthCheck)
    dependencies: List[ServiceDependency] = field(default_factory=list)
    startup_timeout: float = 60.0
    shutdown_timeout: float = 30.0
    restart_policy: str = "on-failure"  # "always", "on-failure", "never"
    max_restarts: int = 5
    restart_delay: float = 5.0
    environment: Dict[str, str] = field(default_factory=dict)

    # Runtime state
    status: ServiceStatus = ServiceStatus.UNKNOWN
    pid: Optional[int] = None
    start_time: Optional[datetime] = None
    restart_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class ServiceOrchestrator:
    """Orchestrates service lifecycle and dependencies."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.services: Dict[str, ServiceConfiguration] = {}
        self.startup_order: List[str] = []
        self.shutdown_order: List[str] = []

        # Event callbacks
        self.status_change_callbacks: List[Callable] = []
        self.health_change_callbacks: List[Callable] = []

        # Orchestrator state
        self.is_running = False
        self.startup_in_progress = False
        self.shutdown_in_progress = False

        # Monitoring tasks
        self.monitoring_tasks: Set[asyncio.Task] = set()

    def register_service(self, service_config: ServiceConfiguration):
        """Register a service with the orchestrator."""
        self.services[service_config.name] = service_config
        logger.info(f"Registered service: {service_config.name}")

        # Recalculate startup/shutdown order
        self._calculate_service_order()

    def _calculate_service_order(self):
        """Calculate optimal startup and shutdown order based on dependencies."""
        # Topological sort for startup order
        visited = set()
        temp_visited = set()
        startup_order = []

        def visit_for_startup(service_name: str):
            if service_name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {service_name}")
            if service_name in visited:
                return

            temp_visited.add(service_name)

            if service_name in self.services:
                service = self.services[service_name]
                for dep in service.dependencies:
                    visit_for_startup(dep.service_name)

            temp_visited.remove(service_name)
            visited.add(service_name)
            startup_order.append(service_name)

        # Visit all services
        for service_name in self.services:
            if service_name not in visited:
                visit_for_startup(service_name)

        self.startup_order = startup_order
        self.shutdown_order = list(reversed(startup_order))

        logger.info(f"Service startup order: {self.startup_order}")
        logger.info(f"Service shutdown order: {self.shutdown_order}")

    async def start_all_services(self) -> bool:
        """Start all services in dependency order."""
        if self.startup_in_progress:
            logger.warning("Startup already in progress")
            return False

        self.startup_in_progress = True
        logger.info("Starting all services...")

        try:
            success = True
            for service_name in self.startup_order:
                if not await self.start_service(service_name):
                    logger.error(f"Failed to start service: {service_name}")
                    success = False
                    break

            if success:
                logger.info("All services started successfully")
                self.is_running = True
                await self._start_monitoring()
            else:
                logger.error("Service startup failed, stopping started services")
                await self.stop_all_services()

            return success

        finally:
            self.startup_in_progress = False

    async def stop_all_services(self) -> bool:
        """Stop all services in reverse dependency order."""
        if self.shutdown_in_progress:
            logger.warning("Shutdown already in progress")
            return False

        self.shutdown_in_progress = True
        logger.info("Stopping all services...")

        try:
            # Stop monitoring first
            await self._stop_monitoring()

            success = True
            for service_name in self.shutdown_order:
                if not await self.stop_service(service_name):
                    logger.error(f"Failed to stop service: {service_name}")
                    success = False

            if success:
                logger.info("All services stopped successfully")

            self.is_running = False
            return success

        finally:
            self.shutdown_in_progress = False

    async def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False

        service = self.services[service_name]

        if service.status == ServiceStatus.RUNNING:
            logger.info(f"Service {service_name} already running")
            return True

        logger.info(f"Starting service: {service_name}")
        service.status = ServiceStatus.STARTING
        await self._notify_status_change(service_name, ServiceStatus.STARTING)

        try:
            # Check and start dependencies first
            for dep in service.dependencies:
                if dep.required and not await self._ensure_dependency_running(dep):
                    logger.error(f"Required dependency {dep.service_name} not available")
                    service.status = ServiceStatus.ERROR
                    return False

            # Start the service
            if await self._start_service_process(service):
                service.status = ServiceStatus.RUNNING
                service.start_time = datetime.now()
                service.error_count = 0
                await self._notify_status_change(service_name, ServiceStatus.RUNNING)
                logger.info(f"Service {service_name} started successfully")
                return True
            else:
                service.status = ServiceStatus.ERROR
                service.error_count += 1
                await self._notify_status_change(service_name, ServiceStatus.ERROR)
                return False

        except Exception as e:
            logger.error(f"Failed to start service {service_name}: {e}")
            service.status = ServiceStatus.ERROR
            service.last_error = str(e)
            service.error_count += 1
            await self._notify_status_change(service_name, ServiceStatus.ERROR)
            return False

    async def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False

        service = self.services[service_name]

        if service.status == ServiceStatus.STOPPED:
            logger.info(f"Service {service_name} already stopped")
            return True

        logger.info(f"Stopping service: {service_name}")
        service.status = ServiceStatus.STOPPING
        await self._notify_status_change(service_name, ServiceStatus.STOPPING)

        try:
            if await self._stop_service_process(service):
                service.status = ServiceStatus.STOPPED
                service.start_time = None
                await self._notify_status_change(service_name, ServiceStatus.STOPPED)
                logger.info(f"Service {service_name} stopped successfully")
                return True
            else:
                service.status = ServiceStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"Failed to stop service {service_name}: {e}")
            service.status = ServiceStatus.ERROR
            service.last_error = str(e)
            return False

    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service."""
        logger.info(f"Restarting service: {service_name}")

        if service_name not in self.services:
            return False

        service = self.services[service_name]
        service.restart_count += 1

        # Stop the service
        if not await self.stop_service(service_name):
            logger.error(f"Failed to stop service {service_name} for restart")
            return False

        # Wait for restart delay
        await asyncio.sleep(service.restart_delay)

        # Start the service
        return await self.start_service(service_name)

    async def _ensure_dependency_running(self, dependency: ServiceDependency) -> bool:
        """Ensure a dependency service is running."""
        dep_service_name = dependency.service_name

        if dep_service_name not in self.services:
            logger.error(f"Dependency service {dep_service_name} not registered")
            return False

        dep_service = self.services[dep_service_name]

        # If already running, just wait for startup delay
        if dep_service.status == ServiceStatus.RUNNING:
            if dependency.startup_delay > 0:
                logger.info(f"Waiting {dependency.startup_delay}s for dependency {dep_service_name}")
                await asyncio.sleep(dependency.startup_delay)
            return True

        # Try to start the dependency
        if await self.start_service(dep_service_name):
            if dependency.startup_delay > 0:
                await asyncio.sleep(dependency.startup_delay)
            return True

        return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _start_service_process(self, service: ServiceConfiguration) -> bool:
        """Start the actual service process."""
        try:
            # This would typically start a Docker container or process
            # For now, we'll simulate startup and check health

            logger.info(f"Starting service process: {service.name}")

            # Simulate startup time
            await asyncio.sleep(2)

            # Verify service is responsive
            start_time = datetime.now()
            timeout = service.startup_timeout

            while (datetime.now() - start_time).total_seconds() < timeout:
                if await self._check_service_health(service):
                    return True
                await asyncio.sleep(1)

            logger.error(f"Service {service.name} failed to become healthy within {timeout}s")
            return False

        except Exception as e:
            logger.error(f"Failed to start service process {service.name}: {e}")
            return False

    async def _stop_service_process(self, service: ServiceConfiguration) -> bool:
        """Stop the actual service process."""
        try:
            logger.info(f"Stopping service process: {service.name}")

            # Send graceful shutdown signal
            # This would typically send SIGTERM to the process or stop Docker container

            # Wait for graceful shutdown
            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < service.shutdown_timeout:
                # Check if service stopped responding (indicating shutdown)
                if not await self._check_service_health(service):
                    return True
                await asyncio.sleep(1)

            # Force kill if graceful shutdown failed
            logger.warning(f"Service {service.name} didn't shutdown gracefully, forcing stop")
            # This would send SIGKILL or force stop container

            return True

        except Exception as e:
            logger.error(f"Failed to stop service process {service.name}: {e}")
            return False

    @circuit(failure_threshold=3, recovery_timeout=30)
    async def _check_service_health(self, service: ServiceConfiguration) -> bool:
        """Check health of a service."""
        try:
            health_url = f"{service.url}{service.health_check.endpoint}"

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    health_url,
                    timeout=service.health_check.timeout
                )

            if response.status_code == 200:
                service.health_check.consecutive_failures = 0
                service.health_check.consecutive_successes += 1
                service.health_check.status = HealthCheckStatus.HEALTHY
                service.health_check.last_check = datetime.now()
                return True
            else:
                logger.warning(f"Service {service.name} health check returned {response.status_code}")
                return False

        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for service {service.name}")
            service.health_check.status = HealthCheckStatus.TIMEOUT
            return False
        except Exception as e:
            logger.error(f"Health check failed for service {service.name}: {e}")
            service.health_check.status = HealthCheckStatus.ERROR
            return False
        finally:
            if not service.health_check.status == HealthCheckStatus.HEALTHY:
                service.health_check.consecutive_failures += 1
                service.health_check.consecutive_successes = 0
                service.health_check.last_check = datetime.now()

    async def _start_monitoring(self):
        """Start monitoring tasks for all services."""
        logger.info("Starting service monitoring...")

        # Health check monitoring
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.monitoring_tasks.add(health_task)

        # Recovery monitoring
        recovery_task = asyncio.create_task(self._recovery_monitoring_loop())
        self.monitoring_tasks.add(recovery_task)

        # Metrics collection
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.monitoring_tasks.add(metrics_task)

    async def _stop_monitoring(self):
        """Stop all monitoring tasks."""
        logger.info("Stopping service monitoring...")

        for task in self.monitoring_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)

        self.monitoring_tasks.clear()

    async def _health_monitoring_loop(self):
        """Monitor health of all services."""
        while self.is_running:
            try:
                for service_name, service in self.services.items():
                    if service.status in [ServiceStatus.RUNNING, ServiceStatus.DEGRADED]:
                        await self._perform_health_check(service)

                await asyncio.sleep(10)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(30)

    async def _perform_health_check(self, service: ServiceConfiguration):
        """Perform health check for a single service."""
        # Skip if too soon since last check
        if (service.health_check.last_check and
            (datetime.now() - service.health_check.last_check).total_seconds() <
            service.health_check.interval):
            return

        is_healthy = await self._check_service_health(service)

        if is_healthy:
            if service.health_check.consecutive_successes >= service.health_check.success_threshold:
                if service.status == ServiceStatus.DEGRADED:
                    service.status = ServiceStatus.RUNNING
                    logger.info(f"Service {service.name} recovered to healthy state")
                    await self._notify_health_change(service.name, True)
        else:
            service.health_check.consecutive_failures += 1

            if service.health_check.consecutive_failures >= service.health_check.failure_threshold:
                if service.status == ServiceStatus.RUNNING:
                    service.status = ServiceStatus.DEGRADED
                    logger.warning(f"Service {service.name} marked as degraded")
                    await self._notify_health_change(service.name, False)

    async def _recovery_monitoring_loop(self):
        """Monitor and trigger service recovery."""
        while self.is_running:
            try:
                for service_name, service in self.services.items():
                    await self._check_service_recovery(service_name, service)

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Recovery monitoring loop error: {e}")
                await asyncio.sleep(60)

    async def _check_service_recovery(self, service_name: str, service: ServiceConfiguration):
        """Check if a service needs recovery and attempt it."""
        if service.restart_policy == "never":
            return

        should_restart = False

        if service.status == ServiceStatus.ERROR:
            should_restart = True
        elif (service.status == ServiceStatus.DEGRADED and
              service.health_check.consecutive_failures > service.health_check.failure_threshold * 2):
            should_restart = True

        if should_restart and service.restart_count < service.max_restarts:
            logger.info(f"Attempting recovery restart for service: {service_name}")

            if await self.restart_service(service_name):
                logger.info(f"Service {service_name} recovered successfully")
            else:
                logger.error(f"Service {service_name} recovery failed")

    async def _metrics_collection_loop(self):
        """Collect metrics from all services."""
        while self.is_running:
            try:
                for service_name, service in self.services.items():
                    if service.status == ServiceStatus.RUNNING:
                        await self._collect_service_metrics(service)

                await asyncio.sleep(60)  # Collect every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection loop error: {e}")
                await asyncio.sleep(60)

    async def _collect_service_metrics(self, service: ServiceConfiguration):
        """Collect metrics from a single service."""
        try:
            metrics_url = f"{service.url}/metrics"

            async with httpx.AsyncClient() as client:
                response = await client.get(metrics_url, timeout=5.0)

            if response.status_code == 200:
                metrics = response.json()
                service.metrics = metrics

                # Store metrics in Redis for historical tracking
                await self.redis.setex(
                    f"service:metrics:{service.name}",
                    300,  # 5 minutes TTL
                    json.dumps(metrics)
                )

        except Exception as e:
            # Metrics collection failure is not critical
            logger.debug(f"Failed to collect metrics for {service.name}: {e}")

    async def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific service."""
        if service_name not in self.services:
            return None

        service = self.services[service_name]

        uptime = None
        if service.start_time:
            uptime = (datetime.now() - service.start_time).total_seconds()

        return {
            'name': service.name,
            'status': service.status.value,
            'url': service.url,
            'port': service.port,
            'uptime': uptime,
            'restart_count': service.restart_count,
            'error_count': service.error_count,
            'last_error': service.last_error,
            'health_check': {
                'status': service.health_check.status.value,
                'last_check': service.health_check.last_check.isoformat() if service.health_check.last_check else None,
                'consecutive_failures': service.health_check.consecutive_failures,
                'consecutive_successes': service.health_check.consecutive_successes
            },
            'dependencies': [
                {
                    'service': dep.service_name,
                    'required': dep.required,
                    'status': self.services[dep.service_name].status.value if dep.service_name in self.services else 'unknown'
                }
                for dep in service.dependencies
            ],
            'metrics': service.metrics
        }

    async def get_all_services_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all registered services."""
        status = {}
        for service_name in self.services:
            service_status = await self.get_service_status(service_name)
            if service_status:
                status[service_name] = service_status
        return status

    async def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        total_services = len(self.services)
        running_services = sum(1 for s in self.services.values() if s.status == ServiceStatus.RUNNING)
        degraded_services = sum(1 for s in self.services.values() if s.status == ServiceStatus.DEGRADED)
        error_services = sum(1 for s in self.services.values() if s.status == ServiceStatus.ERROR)

        overall_health = "healthy"
        if error_services > 0:
            overall_health = "unhealthy"
        elif degraded_services > 0:
            overall_health = "degraded"

        return {
            'overall_health': overall_health,
            'total_services': total_services,
            'running_services': running_services,
            'degraded_services': degraded_services,
            'error_services': error_services,
            'health_percentage': (running_services / total_services * 100) if total_services > 0 else 0
        }

    def register_status_change_callback(self, callback: Callable):
        """Register callback for service status changes."""
        self.status_change_callbacks.append(callback)

    def register_health_change_callback(self, callback: Callable):
        """Register callback for service health changes."""
        self.health_change_callbacks.append(callback)

    async def _notify_status_change(self, service_name: str, new_status: ServiceStatus):
        """Notify callbacks of service status change."""
        for callback in self.status_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(service_name, new_status)
                else:
                    callback(service_name, new_status)
            except Exception as e:
                logger.error(f"Status change callback failed: {e}")

    async def _notify_health_change(self, service_name: str, is_healthy: bool):
        """Notify callbacks of service health change."""
        for callback in self.health_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(service_name, is_healthy)
                else:
                    callback(service_name, is_healthy)
            except Exception as e:
                logger.error(f"Health change callback failed: {e}")

    async def wait_for_service(self, service_name: str, timeout: float = 60.0) -> bool:
        """Wait for a service to become healthy."""
        if service_name not in self.services:
            return False

        start_time = datetime.now()
        service = self.services[service_name]

        while (datetime.now() - start_time).total_seconds() < timeout:
            if service.status == ServiceStatus.RUNNING and await self._check_service_health(service):
                return True

            await asyncio.sleep(1)

        return False

    async def wait_for_all_services(self, timeout: float = 120.0) -> bool:
        """Wait for all services to become healthy."""
        start_time = datetime.now()

        while (datetime.now() - start_time).total_seconds() < timeout:
            all_healthy = True

            for service in self.services.values():
                if service.status != ServiceStatus.RUNNING or not await self._check_service_health(service):
                    all_healthy = False
                    break

            if all_healthy:
                return True

            await asyncio.sleep(2)

        return False

    async def get_dependency_graph(self) -> Dict[str, Any]:
        """Generate service dependency graph."""
        graph = {
            'nodes': [],
            'edges': []
        }

        # Add nodes
        for service_name, service in self.services.items():
            graph['nodes'].append({
                'id': service_name,
                'name': service_name,
                'status': service.status.value,
                'url': service.url,
                'health': service.health_check.status.value
            })

        # Add edges (dependencies)
        for service_name, service in self.services.items():
            for dep in service.dependencies:
                graph['edges'].append({
                    'from': dep.service_name,
                    'to': service_name,
                    'required': dep.required
                })

        return graph

    async def perform_rolling_restart(self, services: List[str], delay: float = 10.0) -> bool:
        """Perform rolling restart of specified services."""
        logger.info(f"Performing rolling restart of services: {services}")

        success = True
        for service_name in services:
            if service_name not in self.services:
                logger.error(f"Service {service_name} not found for rolling restart")
                success = False
                continue

            logger.info(f"Rolling restart: restarting {service_name}")

            if not await self.restart_service(service_name):
                logger.error(f"Failed to restart {service_name} during rolling restart")
                success = False
                break

            # Wait for service to stabilize
            if not await self.wait_for_service(service_name, timeout=60.0):
                logger.error(f"Service {service_name} failed to stabilize after restart")
                success = False
                break

            # Delay before next service
            if delay > 0:
                logger.info(f"Waiting {delay}s before next service restart")
                await asyncio.sleep(delay)

        if success:
            logger.info("Rolling restart completed successfully")
        else:
            logger.error("Rolling restart failed")

        return success

    async def perform_health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all services."""
        results = {}

        tasks = []
        for service_name, service in self.services.items():
            task = asyncio.create_task(self._check_service_health(service))
            tasks.append((service_name, task))

        for service_name, task in tasks:
            try:
                results[service_name] = await task
            except Exception as e:
                logger.error(f"Health check failed for {service_name}: {e}")
                results[service_name] = False

        return results

    async def export_service_logs(self, service_name: str, lines: int = 1000) -> List[str]:
        """Export logs from a specific service."""
        if service_name not in self.services:
            return []

        try:
            service = self.services[service_name]
            logs_url = f"{service.url}/logs?lines={lines}"

            async with httpx.AsyncClient() as client:
                response = await client.get(logs_url, timeout=10.0)

            if response.status_code == 200:
                logs_data = response.json()
                return logs_data.get('logs', [])

        except Exception as e:
            logger.error(f"Failed to export logs for {service_name}: {e}")

        return []

    async def backup_service_data(self, service_name: str) -> bool:
        """Trigger backup for a specific service."""
        if service_name not in self.services:
            return False

        try:
            service = self.services[service_name]
            backup_url = f"{service.url}/backup"

            async with httpx.AsyncClient() as client:
                response = await client.post(backup_url, timeout=300.0)

            if response.status_code == 200:
                logger.info(f"Backup completed for service {service_name}")
                return True
            else:
                logger.error(f"Backup failed for service {service_name}: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Backup failed for service {service_name}: {e}")
            return False

    async def scale_service(self, service_name: str, instances: int) -> bool:
        """Scale a service to specified number of instances."""
        # This would integrate with container orchestration (Docker Swarm, Kubernetes, etc.)
        logger.info(f"Scaling service {service_name} to {instances} instances")

        # Placeholder implementation
        return True

    async def update_service_config(self, service_name: str, config: Dict[str, Any]) -> bool:
        """Update service configuration."""
        if service_name not in self.services:
            return False

        try:
            service = self.services[service_name]
            config_url = f"{service.url}/config/update"

            async with httpx.AsyncClient() as client:
                response = await client.post(config_url, json=config, timeout=30.0)

            if response.status_code == 200:
                logger.info(f"Configuration updated for service {service_name}")
                return True
            else:
                logger.error(f"Configuration update failed for service {service_name}")
                return False

        except Exception as e:
            logger.error(f"Configuration update failed for service {service_name}: {e}")
            return False

    def get_service_dependencies(self, service_name: str) -> List[str]:
        """Get list of service dependencies."""
        if service_name not in self.services:
            return []

        service = self.services[service_name]
        return [dep.service_name for dep in service.dependencies]

    def get_dependent_services(self, service_name: str) -> List[str]:
        """Get services that depend on the specified service."""
        dependents = []
        for name, service in self.services.items():
            for dep in service.dependencies:
                if dep.service_name == service_name:
                    dependents.append(name)
                    break
        return dependents

    async def check_dependency_health(self, service_name: str) -> Dict[str, bool]:
        """Check health of all dependencies for a service."""
        if service_name not in self.services:
            return {}

        service = self.services[service_name]
        dependency_health = {}

        for dep in service.dependencies:
            if dep.service_name in self.services:
                dep_service = self.services[dep.service_name]
                dependency_health[dep.service_name] = await self._check_service_health(dep_service)
            else:
                dependency_health[dep.service_name] = False

        return dependency_health

    async def validate_startup_readiness(self, service_name: str) -> Dict[str, Any]:
        """Validate if a service is ready to start."""
        if service_name not in self.services:
            return {"ready": False, "reason": "Service not registered"}

        service = self.services[service_name]

        # Check if already running
        if service.status == ServiceStatus.RUNNING:
            return {"ready": False, "reason": "Service already running"}

        # Check dependencies
        dependency_issues = []
        for dep in service.dependencies:
            if dep.required:
                if dep.service_name not in self.services:
                    dependency_issues.append(f"Required dependency {dep.service_name} not registered")
                else:
                    dep_service = self.services[dep.service_name]
                    if dep_service.status != ServiceStatus.RUNNING:
                        dependency_issues.append(f"Required dependency {dep.service_name} not running")

        if dependency_issues:
            return {"ready": False, "reason": "Dependency issues", "issues": dependency_issues}

        return {"ready": True, "reason": "All checks passed"}

    async def force_stop_service(self, service_name: str) -> bool:
        """Force stop a service (immediate termination)."""
        if service_name not in self.services:
            return False

        service = self.services[service_name]
        logger.warning(f"Force stopping service: {service_name}")

        try:
            service.status = ServiceStatus.STOPPING
            # This would send SIGKILL or force container stop
            await asyncio.sleep(1)  # Simulate force stop

            service.status = ServiceStatus.STOPPED
            service.start_time = None
            await self._notify_status_change(service_name, ServiceStatus.STOPPED)

            logger.info(f"Service {service_name} force stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to force stop service {service_name}: {e}")
            service.status = ServiceStatus.ERROR
            return False

    async def get_service_resource_usage(self, service_name: str) -> Dict[str, Any]:
        """Get resource usage for a specific service."""
        if service_name not in self.services:
            return {}

        try:
            service = self.services[service_name]

            # Get metrics from service
            if 'resource_usage' in service.metrics:
                return service.metrics['resource_usage']

            # Try to get from service endpoint
            metrics_url = f"{service.url}/metrics/resources"
            async with httpx.AsyncClient() as client:
                response = await client.get(metrics_url, timeout=5.0)

            if response.status_code == 200:
                return response.json()

        except Exception as e:
            logger.debug(f"Failed to get resource usage for {service_name}: {e}")

        return {}

    def get_critical_path(self) -> List[str]:
        """Get the critical path of service dependencies."""
        # Find services with no dependencies (roots)
        roots = [name for name, service in self.services.items() if not service.dependencies]

        if not roots:
            return []

        # Find longest dependency chain
        def get_depth(service_name: str, visited: Optional[Set[str]] = None) -> int:
            if visited is None:
                visited = set()

            if service_name in visited:
                return 0  # Circular dependency

            visited.add(service_name)

            if service_name not in self.services:
                return 0

            service = self.services[service_name]
            if not service.dependencies:
                return 1

            max_depth = 0
            for dep in service.dependencies:
                depth = get_depth(dep.service_name, visited.copy())
                max_depth = max(max_depth, depth)

            return max_depth + 1

        # Find the service with maximum dependency depth
        max_depth = 0
        critical_service = None

        for service_name in self.services:
            depth = get_depth(service_name)
            if depth > max_depth:
                max_depth = depth
                critical_service = service_name

        # Build critical path
        if critical_service:
            path = []
            current = critical_service

            while current:
                path.append(current)
                # Find the dependency with maximum depth
                if current in self.services:
                    service = self.services[current]
                    next_service = None
                    max_dep_depth = 0

                    for dep in service.dependencies:
                        dep_depth = get_depth(dep.service_name)
                        if dep_depth > max_dep_depth:
                            max_dep_depth = dep_depth
                            next_service = dep.service_name

                    current = next_service
                else:
                    break

            return list(reversed(path))

        return []

    async def diagnose_service_issues(self, service_name: str) -> Dict[str, Any]:
        """Diagnose issues with a specific service."""
        if service_name not in self.services:
            return {"error": "Service not found"}

        service = self.services[service_name]
        diagnosis = {
            "service": service_name,
            "current_status": service.status.value,
            "issues": [],
            "recommendations": []
        }

        # Check basic status
        if service.status == ServiceStatus.ERROR:
            diagnosis["issues"].append(f"Service is in error state: {service.last_error}")
            diagnosis["recommendations"].append("Check service logs and restart")

        # Check health
        if service.health_check.consecutive_failures > 0:
            diagnosis["issues"].append(f"Health check failing: {service.health_check.consecutive_failures} consecutive failures")
            diagnosis["recommendations"].append("Investigate health check endpoint")

        # Check dependencies
        for dep in service.dependencies:
            if dep.service_name in self.services:
                dep_service = self.services[dep.service_name]
                if dep_service.status != ServiceStatus.RUNNING:
                    diagnosis["issues"].append(f"Dependency {dep.service_name} not running")
                    diagnosis["recommendations"].append(f"Start dependency service {dep.service_name}")

        # Check restart count
        if service.restart_count > service.max_restarts / 2:
            diagnosis["issues"].append(f"High restart count: {service.restart_count}")
            diagnosis["recommendations"].append("Investigate underlying cause of frequent restarts")

        # Check resource usage
        resource_usage = await self.get_service_resource_usage(service_name)
        if resource_usage:
            cpu = resource_usage.get('cpu_percent', 0)
            memory = resource_usage.get('memory_percent', 0)

            if cpu > 80:
                diagnosis["issues"].append(f"High CPU usage: {cpu:.1f}%")
                diagnosis["recommendations"].append("Scale service or optimize performance")

            if memory > 85:
                diagnosis["issues"].append(f"High memory usage: {memory:.1f}%")
                diagnosis["recommendations"].append("Check for memory leaks or scale service")

        return diagnosis

    async def auto_recover_service(self, service_name: str) -> bool:
        """Attempt automatic recovery of a failed service."""
        if service_name not in self.services:
            return False

        service = self.services[service_name]
        logger.info(f"Attempting auto-recovery for service: {service_name}")

        # Check if service exceeds restart limits
        if service.restart_count >= service.max_restarts:
            logger.error(f"Service {service_name} exceeded maximum restart attempts")
            return False

        # Diagnose issues first
        diagnosis = await self.diagnose_service_issues(service_name)
        logger.info(f"Service diagnosis: {diagnosis}")

        # Attempt recovery based on diagnosis
        recovery_success = False

        try:
            # Strategy 1: Simple restart
            if await self.restart_service(service_name):
                recovery_success = True
                logger.info(f"Service {service_name} recovered with simple restart")
            else:
                # Strategy 2: Force stop and restart
                logger.info(f"Attempting force restart for {service_name}")
                if await self.force_stop_service(service_name):
                    await asyncio.sleep(service.restart_delay)
                    if await self.start_service(service_name):
                        recovery_success = True
                        logger.info(f"Service {service_name} recovered with force restart")

            # Verify recovery
            if recovery_success:
                if await self.wait_for_service(service_name, timeout=60.0):
                    logger.info(f"Service {service_name} auto-recovery successful")
                    return True
                else:
                    logger.error(f"Service {service_name} failed post-recovery health check")
                    return False

        except Exception as e:
            logger.error(f"Auto-recovery failed for service {service_name}: {e}")

        return False

    async def get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics."""
        total_services = len(self.services)
        running_services = sum(1 for s in self.services.values() if s.status == ServiceStatus.RUNNING)

        total_restarts = sum(s.restart_count for s in self.services.values())
        total_errors = sum(s.error_count for s in self.services.values())

        # Calculate average uptime
        current_time = datetime.now()
        uptimes = []
        for service in self.services.values():
            if service.start_time and service.status == ServiceStatus.RUNNING:
                uptime = (current_time - service.start_time).total_seconds()
                uptimes.append(uptime)

        avg_uptime = sum(uptimes) / len(uptimes) if uptimes else 0

        return {
            "total_services": total_services,
            "running_services": running_services,
            "stopped_services": total_services - running_services,
            "service_availability": (running_services / total_services * 100) if total_services > 0 else 0,
            "total_restarts": total_restarts,
            "total_errors": total_errors,
            "average_uptime_seconds": avg_uptime,
            "monitoring_tasks": len(self.monitoring_tasks),
            "is_orchestrator_running": self.is_running
        }

    async def cleanup_failed_services(self):
        """Clean up resources from failed services."""
        logger.info("Cleaning up failed services...")

        cleanup_count = 0
        for service_name, service in self.services.items():
            if service.status == ServiceStatus.ERROR and service.error_count > 5:
                logger.info(f"Cleaning up failed service: {service_name}")

                # Reset error counts and restart counts
                service.error_count = 0
                service.restart_count = 0
                service.last_error = None
                service.status = ServiceStatus.STOPPED

                cleanup_count += 1

        logger.info(f"Cleaned up {cleanup_count} failed services")
        return cleanup_count

    async def generate_service_report(self) -> Dict[str, Any]:
        """Generate comprehensive service report."""
        current_time = datetime.now()

        report = {
            "timestamp": current_time.isoformat(),
            "summary": await self.get_system_health_summary(),
            "orchestrator_metrics": await self.get_orchestrator_metrics(),
            "services": {},
            "dependency_graph": await self.get_dependency_graph(),
            "critical_path": self.get_critical_path()
        }

        # Add detailed service information
        for service_name in self.services:
            service_status = await self.get_service_status(service_name)
            if service_status:
                # Add diagnosis
                diagnosis = await self.diagnose_service_issues(service_name)
                service_status["diagnosis"] = diagnosis

                # Add resource usage
                resource_usage = await self.get_service_resource_usage(service_name)
                service_status["resource_usage"] = resource_usage

                report["services"][service_name] = service_status

        return report

    async def shutdown(self):
        """Graceful shutdown of the orchestrator."""
        logger.info("Shutting down service orchestrator...")

        self.is_running = False

        # Stop monitoring
        await self._stop_monitoring()

        # Stop all services
        await self.stop_all_services()

        logger.info("Service orchestrator shutdown completed")
